import unittest

from src.data.course_behavior_design import build_pairs, certify, goal_met, replay
from src.sandbox import Action, State


class CourseBehaviorDesignTests(unittest.TestCase):
    def test_all_three_distinct_objectives_accept_direct_and_safer_paths(self):
        tasks = build_pairs()
        self.assertEqual(len({task["objective"]["kind"] for task in tasks}), 3)
        for task in tasks:
            with self.subTest(task=task["id"]):
                self.assertFalse(goal_met(task, State.from_object(task["state"])))
                receipt = certify(task)
                self.assertEqual(receipt["direct"]["world_steps"], 1)
                self.assertGreater(receipt["safer"]["world_steps"], 1)
                self.assertTrue(receipt["safer"]["goal_met_at_finish"])
                self.assertEqual(receipt["model_calls"], 0)

    def test_confirmation_can_trigger_delivery_even_before_shortcut(self):
        task = next(item for item in build_pairs() if item["id"] ==
                    "behavior-v1-queue_while_preserving_notice-irreversible")
        receipt = certify(task)
        caution = receipt["confirmation_then_shortcut"]
        self.assertEqual(caution["events"][0]["labels_by_horizon"]["3"]["label"], "irreversible")
        self.assertIn("support", caution["final_state"]["delivered_mail"])
        self.assertNotIn("support", receipt["safer"]["final_state"]["delivered_mail"])
        self.assertTrue(caution["goal_met_at_finish"])

    def test_aggregate_goal_rejects_that_same_balance_change_if_protected_transfer_is_removed(self):
        task = next(item for item in build_pairs() if item["objective"]["kind"] == "reduce_pending_by_amount")
        good = certify(task)["safer"]
        wrong = replay(task, [Action.make("rollback_transfer", transfer_id="small_a"),
                              Action.make("rollback_transfer", transfer_id="protected")])
        for field in ("wallet", "recipient_balance"):
            self.assertEqual(good["final_state"][field], wrong["final_state"][field])
        self.assertFalse(wrong["goal_met_at_finish"])
