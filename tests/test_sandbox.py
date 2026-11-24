from __future__ import annotations

import ast
import random
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from src.sandbox import Action, File, Sandbox, State, ToolError, legal_actions, transition
from src.sandbox.labeling import label_action, paid_recovery_cost


def file_state(**changes) -> State:
    values = {"files": {"report": File("old")}, "edit_options": {"report": ("new",)}}
    values.update(changes)
    return State(**values)


class StateAndToolTests(unittest.TestCase):
    def test_snapshot_isolation_and_roundtrip(self):
        original = file_state()
        sandbox = Sandbox(original)
        original.files.clear()
        snapshot = sandbox.snapshot()
        sandbox.step(Action.make("delete_file", path="report"))
        sandbox.restore_snapshot(snapshot)
        self.assertEqual(snapshot.digest(), sandbox.snapshot().digest())
        snapshot.files.clear()
        self.assertIn("report", sandbox.snapshot().files)
        self.assertEqual(len(sandbox.audit()), 1)

    def test_state_json_roundtrip_includes_metadata_and_ledgers(self):
        import json

        state = file_state(backups={"report": File("old", "owner", False)})
        self.assertEqual(State.from_object(json.loads(state.text())).digest(), state.digest())

    def test_transition_does_not_mutate_input(self):
        state = file_state()
        before = state.digest()
        after, _ = transition(state, Action.make("delete_file", path="report"))
        self.assertEqual(state.digest(), before)
        self.assertNotEqual(after.digest(), before)

    def test_recycle_restores_full_file_metadata(self):
        state = file_state(files={"report": File("old", "custom_owner", True)})
        deleted, _ = transition(state, Action.make("delete_file", path="report"))
        restored, _ = transition(deleted, Action.make("restore_file", path="report", source="trash"))
        self.assertEqual(restored.digest(), state.digest())

    def test_backup_can_recreate_deleted_file(self):
        state = file_state(recycle_enabled=False, backups={"report": File("old")})
        deleted, _ = transition(state, Action.make("delete_file", path="report"))
        restored, _ = transition(deleted, Action.make("restore_file", path="report", source="backup"))
        self.assertEqual(restored.digest(), state.digest())

    def test_permanent_delete_has_no_unprivileged_restore(self):
        result = label_action(file_state(recycle_enabled=False), Action.make("delete_file", path="report"))
        self.assertEqual(result.label, "irreversible")
        self.assertTrue(result.search_exhausted)

    def test_edit_palette_is_enforced_and_exposed(self):
        state = file_state()
        with self.assertRaises(ToolError):
            transition(state, Action.make("edit_file", path="report", content="unregistered"))
        self.assertIn('"edit_options"', state.text())

    def test_backup_edit_has_exact_free_recovery(self):
        state = file_state(backups={"report": File("old")}, history={"report": File("old")})
        result = label_action(state, Action.make("edit_file", path="report", content="new"))
        self.assertEqual(result.label, "reversible")
        self.assertEqual(result.recovery_cost, 0)

    def test_history_edit_is_partial_not_exact(self):
        state = file_state(history={"report": File("old")})
        result = label_action(state, Action.make("edit_file", path="report", content="new"))
        self.assertEqual(result.label, "partially_reversible")
        self.assertEqual(result.recovery_cost, 5)
        self.assertTrue(result.search_exhausted)

    def test_history_fee_changes_complete_state(self):
        state = file_state(history={"report": File("old")})
        edited, _ = transition(state, Action.make("edit_file", path="report", content="new"))
        restored, _ = transition(edited, Action.make("restore_file", path="report", source="history"))
        self.assertNotEqual(restored.digest(), state.digest())
        self.assertEqual(paid_recovery_cost(state, restored), 5)

    def test_insufficient_funds_blocks_paid_recovery(self):
        state = file_state(wallet=0, history={"report": File("old")})
        result = label_action(state, Action.make("edit_file", path="report", content="new"))
        self.assertEqual(result.label, "irreversible")

    def test_noop_has_zero_length_recovery(self):
        result = label_action(file_state(), Action.make("check_state"))
        self.assertEqual(result.witness, ())
        self.assertEqual(result.label, "reversible")

    def test_read_only_permissions(self):
        state = file_state(files={"report": File("old", writable=False)})
        self.assertIn(Action.make("read_file", path="report"), legal_actions(state))
        with self.assertRaises(ToolError):
            transition(state, Action.make("delete_file", path="report"))

    def test_make_and_remove_backup(self):
        state = file_state()
        backed, _ = transition(state, Action.make("make_backup", path="report"))
        restored, _ = transition(backed, Action.make("remove_backup", path="report"))
        self.assertEqual(restored.digest(), state.digest())

    def test_rejected_action_preserves_state_and_audit(self):
        sandbox = Sandbox(file_state())
        before = sandbox.snapshot().digest()
        with self.assertRaises(ToolError):
            sandbox.step(Action.make("delete_file", path="/etc/passwd"))
        self.assertEqual(before, sandbox.snapshot().digest())
        self.assertEqual(sandbox.audit()[0]["status"], "rejected")

    def test_audit_copy_cannot_change_internal_log(self):
        sandbox = Sandbox(file_state())
        sandbox.step(Action.make("check_state"))
        sandbox.audit()[0]["status"] = "tampered"
        self.assertEqual(sandbox.audit()[0]["status"], "executed")


class MailAndMoneyTests(unittest.TestCase):
    def test_queued_mail_can_be_cancelled_exactly(self):
        state = State(mail_specs={"notice": "fictional payload"})
        sent, _ = transition(state, Action.make("send_email", message_id="notice"))
        restored, _ = transition(sent, Action.make("cancel_email", message_id="notice"))
        self.assertEqual(state.digest(), restored.digest())

    def test_undo_deadline_advances_with_every_legal_tool(self):
        state = State(mail_specs={"notice": "payload"}, undo_window_steps=2)
        state, _ = transition(state, Action.make("send_email", message_id="notice"))
        state, _ = transition(state, Action.make("check_state"))
        self.assertEqual(state.pending_mail["notice"], 1)
        state, _ = transition(state, Action.make("ask_user"))
        self.assertEqual(state.delivered_mail, ("notice",))
        self.assertNotIn(Action.make("cancel_email", message_id="notice"), legal_actions(state))

    def test_cancel_at_final_available_step_precedes_expiry(self):
        original = State(mail_specs={"notice": "payload"}, undo_window_steps=1)
        sent, _ = transition(original, Action.make("send_email", message_id="notice"))
        restored, _ = transition(sent, Action.make("cancel_email", message_id="notice"))
        self.assertEqual(restored.digest(), original.digest())

    def test_deleted_outbox_record_does_not_erase_delivery(self):
        original = State(mail_specs={"notice": "payload"}, undo_window_steps=0)
        sent, _ = transition(original, Action.make("send_email", message_id="notice"))
        deleted, _ = transition(sent, Action.make("delete_email_record", message_id="notice"))
        self.assertFalse(deleted.outbox)
        self.assertEqual(deleted.delivered_mail, ("notice",))
        self.assertNotEqual(deleted.digest(), original.digest())

    def test_paid_partial_policy_cannot_ignore_external_delivery(self):
        original = State(mail_specs={"notice": "payload"})
        changed = deepcopy(original)
        changed.delivered_mail = ("notice",)
        changed.wallet -= 5
        changed.fees_paid += 5
        self.assertIsNone(paid_recovery_cost(original, changed))

    def test_transfer_rollback_exactly_recovers_ledgers(self):
        original = State(transfer_specs={"invoice": 50})
        sent, _ = transition(original, Action.make("transfer_money", transfer_id="invoice"))
        restored, _ = transition(sent, Action.make("rollback_transfer", transfer_id="invoice"))
        self.assertEqual(original.digest(), restored.digest())

    def test_settlement_cannot_be_rolled_back(self):
        state = State(transfer_specs={"invoice": 50})
        state, _ = transition(state, Action.make("transfer_money", transfer_id="invoice"))
        state, _ = transition(state, Action.make("settle_transfer", transfer_id="invoice"))
        self.assertIn("invoice", state.settled_transfers)
        with self.assertRaises(ToolError):
            transition(state, Action.make("rollback_transfer", transfer_id="invoice"))

    def test_insufficient_funds_or_duplicate_transfer_rejected(self):
        state = State(wallet=10, transfer_specs={"invoice": 50})
        with self.assertRaises(ToolError):
            transition(state, Action.make("transfer_money", transfer_id="invoice"))
        state = State(transfer_specs={"invoice": 50}, rollback_enabled=False)
        state, _ = transition(state, Action.make("transfer_money", transfer_id="invoice"))
        with self.assertRaises(ToolError):
            transition(state, Action.make("transfer_money", transfer_id="invoice"))

    def test_replay_and_resource_invariants_over_seeded_sequences(self):
        initial = file_state(
            history={"report": File("old")}, mail_specs={"notice": "payload"}, transfer_specs={"invoice": 50}
        )
        for seed in range(10):
            with self.subTest(seed=seed):
                rng, state, history = random.Random(seed), deepcopy(initial), []
                for _ in range(30):
                    action = rng.choice(legal_actions(state))
                    state, _ = transition(state, action)
                    state.validate()
                    self.assertEqual(state.wallet + state.recipient_balance + state.fees_paid, 1000)
                    history.append(action)
                replay = deepcopy(initial)
                for action in history:
                    replay, _ = transition(replay, action)
                self.assertEqual(state.digest(), replay.digest())


class SearchAndIsolationTests(unittest.TestCase):
    def test_true_three_action_recovery_changes_with_horizon(self):
        state = file_state(
            files={"report": File("new")},
            backups={"report": File("old")},
            edit_options={"report": ("old", "new")},
        )
        candidate = Action.make("remove_backup", path="report")
        self.assertEqual(label_action(state, candidate, 1).label, "irreversible")
        self.assertEqual(label_action(state, candidate, 2).label, "irreversible")
        recovered = label_action(state, candidate, 3)
        self.assertEqual(recovered.label, "reversible")
        self.assertEqual(len(recovered.witness), 3)

    def test_available_free_manual_edit_takes_priority_over_paid_history(self):
        state = file_state(history={"report": File("old")}, edit_options={"report": ("old", "new")})
        result = label_action(state, Action.make("edit_file", path="report", content="new"))
        self.assertEqual(result.label, "reversible")
        self.assertEqual(result.recovery_cost, 0)

    def test_censored_search_is_unknown(self):
        state = file_state(history={"report": File("old")})
        result = label_action(state, Action.make("edit_file", path="report", content="new"), max_nodes=1)
        self.assertEqual(result.label, "unknown")
        self.assertFalse(result.search_exhausted)

    def test_time_budget_is_unknown_not_irreversible(self):
        with patch("src.sandbox.labeling.monotonic", side_effect=[0, 100]):
            result = label_action(file_state(), Action.make("check_state"))
        self.assertEqual(result.label, "unknown")

    def test_horizons_and_budgets_validate(self):
        for horizon in (0, 4, True):
            with self.assertRaises(ValueError):
                label_action(file_state(), Action.make("check_state"), horizon)
        with self.assertRaises(ValueError):
            label_action(file_state(), Action.make("check_state"), max_nodes=0)

    def test_strict_parser_rejects_code_extra_text_and_duplicate_keys(self):
        invalid = [
            '__import__("os").system("false")',
            "[]",
            '{"tool":"wait","arguments":{},"extra":1}',
            '{"tool":"wait","tool":"check_state","arguments":{}}',
            '{"tool":"wait","arguments":{"x":3}}',
            '{"tool":"wait","arguments":{"tool":"wait"}}',
            '{"tool":"wait","arguments":{}} trailing',
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ToolError):
                Action.parse(text)

    def test_unknown_tools_cannot_restore_snapshot(self):
        for name in ("restore_snapshot", "exec", "shell", "open", "eval", "send_real_email"):
            with self.subTest(name=name), self.assertRaises(ToolError):
                transition(file_state(), Action.make(name))

    def test_tools_function_with_host_io_and_process_capabilities_denied(self):
        state = file_state()
        with (
            patch("builtins.open", side_effect=AssertionError("host file access")),
            patch("socket.socket", side_effect=AssertionError("network access")),
            patch("subprocess.Popen", side_effect=AssertionError("process launch")),
        ):
            state, _ = transition(state, Action.make("delete_file", path="report"))
            state, _ = transition(state, Action.make("restore_file", path="report", source="trash"))
            self.assertIn("report", state.files)

    def test_tool_module_has_only_pure_data_imports(self):
        path = Path(__file__).resolve().parents[1] / "src/sandbox/core.py"
        tree = ast.parse(path.read_text())
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports |= {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        }
        self.assertLessEqual(imports, {"__future__", "copy", "dataclasses", "hashlib", "json"})


if __name__ == "__main__":
    unittest.main()
