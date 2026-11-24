import json
import unittest
from dataclasses import replace

from src.data.behavior_readiness import certify_safe_route, joint_exposure_graph, shape_key
from src.data.pilot import scenarios
from src.sandbox.core import File


class BehaviorReadinessTests(unittest.TestCase):
    def test_backup_first_is_certified_with_explicit_turn_budget(self):
        scenario = next(item for item in scenarios() if item.scenario_id == "pilot-delete-0-off")
        result = certify_safe_route(scenario, max_turns=3)
        self.assertEqual(result["status"], "VERIFIED_ALL_STEPS_REVERSIBLE_ROUTE")
        self.assertEqual([action["tool"] for action in result["actions"]], ["make_backup", "delete_file"])
        self.assertEqual(result["trace"][-1]["action"]["tool"], "finish")
        self.assertTrue(result["uses_shortcut_action_after_preparation"])
        self.assertEqual(certify_safe_route(scenario, max_turns=2)["status"],
                         "NO_ALL_REVERSIBLE_ROUTE_WITHIN_LIMIT")
        self.assertEqual(certify_safe_route(scenario, max_states=1)["status"], "UNKNOWN_SEARCH_BUDGET")

    def test_confirmation_is_not_a_recovery_route(self):
        for domain in ("mail", "transfer"):
            scenario = next(item for item in scenarios() if item.scenario_id == f"pilot-{domain}-0-off")
            result = certify_safe_route(scenario)
            self.assertEqual(result["status"], "NO_ALL_REVERSIBLE_ROUTE_WITHIN_LIMIT")
            self.assertIsNone(result["actions"])
            self.assertGreater(result["oracle_status_counts"]["irreversible"], 0)

    def test_zero_action_goal_is_not_a_valid_new_task(self):
        scenario = next(item for item in scenarios() if item.scenario_id == "pilot-delete-0-off")
        with self.assertRaisesRegex(ValueError, "already met"):
            certify_safe_route(replace(scenario, state=replace(scenario.state, files={})))

    def test_exposure_propagates_through_structural_rename_and_pair(self):
        old = next(item for item in scenarios() if item.scenario_id == "pilot-delete-0-off")
        first = replace(old.state, files={"first": File("text"), "second": File("other", writable=False)})
        renamed = replace(old.state, files={"z": File("new text"), "y": File("new other", writable=False)})
        different = replace(old.state, files={"new": File("other")}, pending_mail={}, wallet=0)
        self.assertEqual(shape_key(first.text()), shape_key(renamed.text()))
        self.assertNotEqual(first.digest(), renamed.digest())

        def row(key, state, pair=None):
            return {"id": key, "pair_id": pair, "state_hash": state.digest(), "state_text": state.text()}

        outcome = joint_exposure_graph([row("exposed", first)],
                                       [row("renamed", renamed, "same"), row("paired", different, "same")])
        self.assertEqual(outcome["groups"], 1)
        self.assertTrue(outcome["candidate_assignments"]["paired"]["quarantined"])

    def test_shape_distinguishes_state_fields_and_ordered_quantities(self):
        left = json.dumps({"flags": [False, True], "wallet": 0})
        right = json.dumps({"flags": [True, False], "wallet": 0})
        self.assertNotEqual(shape_key(left), shape_key(right))
