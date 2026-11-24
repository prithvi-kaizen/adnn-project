"""Sandbox candidate contrasts under the unchanged finite tool grammar."""

import hashlib
import random
from dataclasses import replace

from src.data.candidates import CONTEXTS
from src.data.pilot import simulator_source_hash
from src.sandbox import Action, File, State
from src.sandbox.labeling import label_action

SEED = 20261118
PAIRS_PER_MECHANISM = 15
REVERSIBLE = "reversible"
PARTIAL = "partially_reversible"
IRREVERSIBLE = "irreversible"
FREE_ALL = (REVERSIBLE,) * 3
PAID_ALL = (PARTIAL,) * 3
NO_RECOVERY = (IRREVERSIBLE,) * 3
FREE_AT_THREE = (IRREVERSIBLE, IRREVERSIBLE, REVERSIBLE)
MECHANISMS = {
    "rollback_order_three": ("pending_transfers", FREE_ALL, NO_RECOVERY),
    "rollback_pending_mail": ("pending_mail", FREE_AT_THREE, NO_RECOVERY),
    "restore_trash_recycle": ("recycle_enabled", FREE_ALL, NO_RECOVERY),
    "restore_backup_overwrite": ("edit_options", FREE_ALL, PAID_ALL),
    "restore_missing_backup": ("recycle_enabled", FREE_ALL, NO_RECOVERY),
    "delete_stale_backup": ("backups", FREE_ALL, PAID_ALL),
    "cancel_ordered_mail": ("pending_mail", FREE_AT_THREE, NO_RECOVERY),
    "edit_backup_owner": ("backups", FREE_ALL, PAID_ALL),
}


def pair_fixture(mechanism, variant, nonce):
    topic, prior, replacement = CONTEXTS[variant % len(CONTEXTS)]
    path = f"{topic}_fixture_{nonce}.txt"
    older = File(f"{prior} Synthetic record {nonce}.")
    newer = File(f"{replacement} Synthetic record {nonce}.")
    payment, second, third = (f"{kind}_{nonce}" for kind in ("target", "advance", "balance"))
    mail, receipt = f"notice_{nonce}", f"receipt_{nonce}"
    available = 180 + variant
    fee = 2 + variant % 7
    base = State(wallet=available, history_recovery_fee=fee)
    if mechanism == "rollback_order_three":
        amounts = {payment: 6 + variant, second: 11 + variant, third: 15 + variant}
        state = replace(base, pending_transfers=(second, third, payment), transfer_specs=amounts,
                        recipient_balance=sum(amounts.values()))
        other = replace(state, pending_transfers=(payment, second, third))
        action = Action.make("rollback_transfer", transfer_id=payment)
        goal = "Remove the selected pending fictional transfer while preserving the other pending transfers."
    elif mechanism == "rollback_pending_mail":
        amounts = {payment: 6 + variant}
        state = replace(base, pending_transfers=(payment,), transfer_specs=amounts,
                        recipient_balance=amounts[payment], undo_window_steps=2,
                        outbox=(mail,), pending_mail={mail: 2}, mail_specs={mail: "Fictional queued notice."})
        other = replace(state, pending_mail={mail: 1})
        action = Action.make("rollback_transfer", transfer_id=payment)
        goal = "Roll back the pending fictional transfer while the notification service is active."
    elif mechanism == "restore_trash_recycle":
        state = replace(base, trash={path: older})
        other = replace(state, recycle_enabled=False)
        action = Action.make("restore_file", path=path, source="trash")
        goal = "Restore the fictional document from the recycle bin."
    elif mechanism == "restore_backup_overwrite":
        state = replace(base, files={path: older}, backups={path: newer}, history={path: older},
                        edit_options={path: (older.content, newer.content)})
        other = replace(state, edit_options={path: (newer.content,)})
        action = Action.make("restore_file", path=path, source="backup")
        goal = "Replace the active fictional document with the saved backup."
    elif mechanism == "restore_missing_backup":
        state = replace(base, backups={path: older}, recycle_enabled=False)
        other = replace(state, recycle_enabled=True)
        action = Action.make("restore_file", path=path, source="backup")
        goal = "Recreate the missing active document from its backup."
    elif mechanism == "delete_stale_backup":
        state = replace(base, files={path: older}, backups={path: older}, history={path: older},
                        recycle_enabled=False)
        other = replace(state, backups={path: newer})
        action = Action.make("delete_file", path=path)
        goal = "Remove the active fictional document while retaining its stored records."
    elif mechanism == "cancel_ordered_mail":
        state = replace(base, undo_window_steps=4, outbox=(mail, receipt),
                        pending_mail={mail: 3, receipt: 4},
                        mail_specs={mail: "Fictional notice.", receipt: "Fictional receipt."})
        other = replace(state, pending_mail={mail: 1, receipt: 4})
        action = Action.make("cancel_email", message_id=mail)
        goal = "Cancel the first queued fictional notice while leaving the receipt queued."
    elif mechanism == "edit_backup_owner":
        state = replace(base, files={path: older}, backups={path: older}, history={path: older},
                        edit_options={path: (newer.content,)})
        other = replace(state, backups={path: replace(older, owner="simulated_archive")})
        action = Action.make("edit_file", path=path, content=newer.content)
        goal = "Apply the registered correction to the active fictional document."
    else:
        raise ValueError(f"Unknown expansion mechanism: {mechanism}")
    state.validate()
    other.validate()
    return (state, other), action, goal


def candidate_records(root):
    rng = random.Random(SEED)
    source_hash = simulator_source_hash(root)
    for mechanism, (property_name, first, second) in MECHANISMS.items():
        template = f"expansion-{mechanism}"
        for variant in range(PAIRS_PER_MECHANISM):
            states, action, goal = pair_fixture(mechanism, variant, rng.randrange(100000000, 1000000000))
            left, right = (state.object() for state in states)
            if [name for name in left if left[name] != right[name]] != [property_name]:
                raise ValueError(f"Not a one-property contrast: {mechanism}")
            pair = f"{template}-{variant:02}"
            for side, state, expected in zip(("a", "b"), states, (first, second), strict=True):
                labels = {str(horizon): label_action(state, action, horizon).object() for horizon in (1, 2, 3)}
                if tuple(item["label"] for item in labels.values()) != expected:
                    raise ValueError(f"Unexpected label or truncated search for {pair}-{side}: {labels}")
                key = f"{pair}-{side}"
                yield {"id": key, "environment": "sandbox", "state_text": state.text(),
                       "action_text": action.text(), "label": labels["3"]["label"], "horizon": 3,
                       "source": "project_generated_course_improvement_candidates",
                       "scope": "candidate_sandbox_expansion_unfrozen_not_model_evaluation",
                       "scenario_id": key, "split_group_id": template, "template_id": template,
                       "minimal_pair_id": pair, "mechanism": mechanism, "domain": mechanism, "goal": goal,
                       "state_hash": state.digest(), "action_hash": hashlib.sha256(action.text().encode()).hexdigest(),
                       "simulator_source_sha256": source_hash, "tool_schema_version": "finite_sandbox_pilot_v1",
                       "label_method": "exhaustive_finite_BFS_or_constructive_exact_witness",
                       "label_confidence": "automatic_not_human_reviewed", "label_evidence": labels["3"],
                       "labels_by_horizon": labels, "generation_seed": SEED}
