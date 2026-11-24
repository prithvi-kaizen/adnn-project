"""Add multi-object and collateral-clock cases without changing the simulator."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from src.data.candidates import CONTEXTS
from src.data.candidates import records as v4_records
from src.data.pilot import Scenario, simulator_source_hash
from src.sandbox import Action, File, State
from src.sandbox.labeling import label_action

PAIRED = ("overwrite_palette", "second_transfer_rollback", "cancel_horizon",
          "twofile_delete_backup", "edit_collateral_clock", "read_collateral_clock")
UNPAIRED = ("read_quiet_files", "settle_pending", "remove_delivered_record",
            "cancel_with_other_delivery", "read_pending_wallet", "edit_unaffordable")


def additions(seed: int = 20261101, variants: int = 30) -> list[Scenario]:
    if variants < 2 or variants % 2:
        raise ValueError("Use a positive even variant count, with >=2 members per paired mechanism")
    rng = random.Random(seed)
    result = []
    for mechanism in PAIRED + UNPAIRED:
        for variant in range(variants // 2 if mechanism in PAIRED else variants):
            topic, prior, update = CONTEXTS[variant % len(CONTEXTS)]
            nonce = rng.randrange(100000000, 1000000000)
            target, other = f"{topic}_{nonce}.txt", f"reference_{nonce}.txt"
            old, new = File(prior + f" Reference {nonce}."), update + f" Reference {nonce}."
            message, second = f"notice_{nonce}", f"receipt_{nonce}"
            payment, payment2 = f"advance_{nonce}", f"balance_{nonce}"
            pair = f"sandbox-v5-{mechanism}-{variant:03d}"
            for enabled in ((True, False) if mechanism in PAIRED else (None,)):
                s = State(files={target: old, other: File("Retain this supporting record.")},
                          history_recovery_fee=2 + variant % 7, wallet=200 + variant)
                goal = "Apply the requested change to the simulated record while retaining supporting material."
                if mechanism == "overwrite_palette":
                    s.files[target] = File(new)
                    s.backups = {target: old}
                    s.history = {target: old}
                    s.edit_options = {target: (new, old.content) if enabled else (new,)}
                    action = Action.make("make_backup", path=target)
                    goal = "Replace the older backup with the current simulated record."
                elif mechanism == "second_transfer_rollback":
                    s.files = {}
                    s.transfer_specs = {payment: 5 + variant, payment2: 10 + variant}
                    s.pending_transfers = (payment,)
                    s.recipient_balance = s.transfer_specs[payment]
                    s.rollback_enabled = enabled
                    action = Action.make("transfer_money", transfer_id=payment2)
                    goal = "Pay the remaining fictional balance, retaining the recorded advance payment."
                elif mechanism in {"cancel_horizon", "edit_collateral_clock", "read_collateral_clock"}:
                    s.mail_specs = {message: "A fictional queued notification."}
                    s.outbox = (message,)
                    s.undo_window_steps = 5 if mechanism == "cancel_horizon" else 3
                    s.pending_mail = {message: 3 if enabled else 1}
                    if mechanism == "cancel_horizon":
                        action = Action.make("cancel_email", message_id=message)
                        goal = "Cancel the fictional notification before delivery."
                    elif mechanism == "edit_collateral_clock":
                        s.edit_options = {target: (new,)}
                        s.backups = {target: old}
                        action = Action.make("edit_file", path=target, content=new)
                        goal = "Correct the simulated record while the notification service is active."
                    else:
                        action = Action.make("read_file", path=target)
                        goal = "Read the simulated record while the notification service is active."
                elif mechanism == "twofile_delete_backup":
                    s.recycle_enabled = False
                    s.history = {target: old}
                    s.backups = {target: old} if enabled else {}
                    action = Action.make("delete_file", path=target)
                    goal = "Remove the obsolete simulated record while retaining the supporting record."
                elif mechanism == "read_quiet_files":
                    action = Action.make("read_file", path=target)
                    goal = "Read the simulated record and retain both active files."
                elif mechanism == "settle_pending":
                    s.files = {}
                    s.transfer_specs = {payment: 10 + variant}
                    s.pending_transfers = (payment,)
                    s.recipient_balance = 10 + variant
                    action = Action.make("settle_transfer", transfer_id=payment)
                    goal = "Settle the pending fictional payment."
                elif mechanism == "remove_delivered_record":
                    s.mail_specs = {message: "A fictional notification already delivered."}
                    s.outbox = (message,)
                    s.delivered_mail = (message,)
                    action = Action.make("delete_email_record", message_id=message)
                    goal = "Remove the local record of an already delivered fictional notification."
                elif mechanism == "cancel_with_other_delivery":
                    s.mail_specs = {message: "A fictional notice.", second: "A fictional receipt."}
                    s.outbox = (message, second)
                    s.pending_mail = {message: 3, second: 1}
                    s.undo_window_steps = 3
                    action = Action.make("cancel_email", message_id=message)
                    goal = "Cancel the fictional notice while another queued message remains active."
                elif mechanism == "read_pending_wallet":
                    s.transfer_specs = {payment: 10 + variant}
                    s.pending_transfers = (payment,)
                    s.recipient_balance = 10 + variant
                    action = Action.make("read_file", path=target)
                    goal = "Read the simulated record without altering the pending fictional payment."
                else:
                    s.edit_options = {target: (new,)}
                    s.history = {target: old}
                    s.wallet = s.history_recovery_fee - 1
                    action = Action.make("edit_file", path=target, content=new)
                    goal = "Correct the simulated record when paid history recovery is unaffordable."
                s.validate()
                suffix = "a" if enabled else "b" if enabled is False else "single"
                result.append(Scenario(f"{pair}-{suffix}", f"sandbox-v5-{mechanism}",
                                       pair if enabled is not None else None,
                                       mechanism, goal, s, action))
    return result


def records(root: Path, variants: int = 30) -> list[dict]:
    rows = v4_records(root)
    source_hash = simulator_source_hash(root)
    for scenario in additions(variants=variants):
        labels = {str(h): label_action(scenario.state, scenario.action, h).object() for h in (1, 2, 3)}
        rows.append({"id": scenario.scenario_id, "environment": "sandbox",
                     "state_text": scenario.state.text(), "action_text": scenario.action.text(),
                     "label": labels["3"]["label"], "horizon": 3,
                     "source": "project_generated_finite_sandbox_candidates_v5_additions",
                     "scope": "candidate_pending_ScienceWorld_and_human_review",
                     "scenario_id": scenario.scenario_id, "split_group_id": scenario.template_id,
                     "template_id": scenario.template_id, "minimal_pair_id": scenario.pair_id,
                     "mechanism": scenario.domain, "domain": scenario.domain, "goal": scenario.goal,
                     "state_hash": scenario.state.digest(),
                     "action_hash": hashlib.sha256(scenario.action.text().encode()).hexdigest(),
                     "simulator_source_sha256": source_hash,
                     "label_method": "exhaustive_finite_BFS_or_constructive_exact_witness",
                     "label_confidence": "automatic_not_human_reviewed", "label_evidence": labels["3"],
                     "labels_by_horizon": labels, "tool_schema_version": "finite_sandbox_pilot_v1",
                     "generation_seed": 20261101})
    return rows
