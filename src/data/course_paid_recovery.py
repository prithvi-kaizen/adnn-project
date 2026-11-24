"""Paid-recovery contrasts for the prospective sandbox coverage audit."""

import hashlib
import random
from dataclasses import replace

from src.data.candidates import CONTEXTS
from src.data.course_expansion import IRREVERSIBLE, NO_RECOVERY, PAID_ALL, PARTIAL
from src.data.pilot import simulator_source_hash
from src.sandbox import Action, File, State
from src.sandbox.labeling import label_action

SEED = 20261119
PAIRS_PER_DESIGN = 45
PAID_AT_THREE = (IRREVERSIBLE, IRREVERSIBLE, PARTIAL)
DESIGNS = {
    "edit_history_with_mail_clock": ("pending_mail", PAID_AT_THREE, NO_RECOVERY),
    "history_fee_with_occupied_trash": ("wallet", PAID_ALL, NO_RECOVERY),
    "remove_dormant_backup_via_history": ("recycle_enabled", PAID_AT_THREE, NO_RECOVERY),
    "paid_restore_to_missing_file": ("recycle_enabled", PAID_ALL, NO_RECOVERY),
}


def paid_pair(design, variant, nonce):
    topic, prior, corrected = CONTEXTS[variant % len(CONTEXTS)]
    path = f"{topic}_recovery_{nonce}.txt"
    old = File(f"{prior} Synthetic case {nonce}.")
    new = File(f"{corrected} Synthetic case {nonce}.")
    fee = 3 + variant % 7
    base = State(wallet=180 + variant, history_recovery_fee=fee, history={path: old})
    if design == "edit_history_with_mail_clock":
        mail = f"notice_{nonce}"
        before = replace(base, files={path: old}, edit_options={path: (new.content,)},
                         outbox=(mail,), pending_mail={mail: 2}, undo_window_steps=2,
                         mail_specs={mail: "Synthetic queued notice."})
        other = replace(before, pending_mail={mail: 1})
        action = Action.make("edit_file", path=path, content=new.content)
        goal = "Apply the registered correction while the documented notification service is active."
    elif design == "history_fee_with_occupied_trash":
        before = replace(base, wallet=fee, files={path: old}, backups={path: new}, trash={path: old})
        other = replace(before, wallet=fee - 1)
        action = Action.make("restore_file", path=path, source="backup")
        goal = "Replace the active document with the available backup."
    elif design == "remove_dormant_backup_via_history":
        before = replace(base, backups={path: old}, recycle_enabled=False)
        other = replace(before, recycle_enabled=True)
        action = Action.make("remove_backup", path=path)
        goal = "Remove the backup of the currently inactive document."
    elif design == "paid_restore_to_missing_file":
        before = replace(base, recycle_enabled=False)
        other = replace(before, recycle_enabled=True)
        action = Action.make("restore_file", path=path, source="history")
        goal = "Recreate the inactive document using its paid history source."
    else:
        raise ValueError(f"Unknown paid-recovery contrast: {design}")
    return (before, other), action, goal


def paid_records(root):
    rng = random.Random(SEED)
    source_hash = simulator_source_hash(root)
    for design, (changed, first, second) in DESIGNS.items():
        template = f"paid-recovery-{design}"
        for variant in range(PAIRS_PER_DESIGN):
            states, action, goal = paid_pair(design, variant, rng.randrange(100000000, 1000000000))
            left, right = (state.object() for state in states)
            if [key for key in left if left[key] != right[key]] != [changed]:
                raise ValueError(f"Invalid minimal-pair contrast: {design}")
            pair = f"{template}-{variant:02}"
            for side, state, expected in zip(("a", "b"), states, (first, second), strict=True):
                labels = {str(horizon): label_action(state, action, horizon).object() for horizon in (1, 2, 3)}
                if tuple(item["label"] for item in labels.values()) != expected:
                    raise ValueError(f"Unexpected or censored recovery label for {pair}-{side}")
                key = f"{pair}-{side}"
                yield {"id": key, "environment": "sandbox", "state_text": state.text(),
                       "action_text": action.text(), "label": labels["3"]["label"], "horizon": 3,
                       "source": "project_generated_paid_recovery_candidates",
                       "scope": "candidate_sandbox_paid_recovery_unfrozen_not_model_evaluation",
                       "scenario_id": key, "split_group_id": template, "template_id": template,
                       "minimal_pair_id": pair, "mechanism": design, "domain": design, "goal": goal,
                       "state_hash": state.digest(), "action_hash": hashlib.sha256(action.text().encode()).hexdigest(),
                       "simulator_source_sha256": source_hash, "tool_schema_version": "finite_sandbox_pilot_v1",
                       "label_method": "exhaustive_finite_BFS_or_constructive_exact_witness",
                       "label_confidence": "automatic_not_human_reviewed", "label_evidence": labels["3"],
                       "labels_by_horizon": labels, "generation_seed": SEED}
