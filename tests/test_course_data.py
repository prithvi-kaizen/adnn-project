import json
import unittest
from copy import deepcopy
from pathlib import Path

from src.agent.loop import run_episode, verify_episode
from src.data.course import (
    allocate_course_split,
    check_access,
    development_tasks,
    inherit_split,
    scenario_from_task,
    select_nested_subset,
    split_summary,
)
from src.sandbox import Action
from src.sandbox.labeling import label_action

ROOT = Path(__file__).resolve().parents[1]


class CourseDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in
                    (ROOT / "data/candidates/v5/sandbox_pairs.jsonl").read_text().splitlines()]
        cls.parent = allocate_course_split(cls.rows, select_nested_subset(cls.rows), 1729)
        cls.tasks, cls.audit = development_tasks(cls.rows)

    def test_subset_is_order_invariant_and_preserves_pairs(self):
        a = select_nested_subset(self.rows)
        b = select_nested_subset(list(reversed(self.rows)))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 360)
        selected = {r["id"] for r in a}
        pairs = {r["minimal_pair_id"] for r in a if r.get("minimal_pair_id")}
        self.assertTrue(all(r["id"] in selected for r in self.rows if r.get("minimal_pair_id") in pairs))

    def test_selection_does_not_use_labels(self):
        original = [r["id"] for r in select_nested_subset(self.rows)]
        relabeled = [dict(r, label="unknown") for r in self.rows]
        self.assertEqual(original, [r["id"] for r in select_nested_subset(relabeled)])

    def test_inherited_subset_split_has_exact_test_fraction_and_group_coverage(self):
        subset = select_nested_subset(self.rows)
        split = inherit_split(subset, self.parent, "test-hash")
        self.assertEqual(split["counts"]["test"], 72)
        self.assertEqual(sum(split["counts"].values()), 360)
        self.assertTrue(all(s["components"] >= 3 for s in split["summary"].values()))
        self.assertEqual(split["component_count"], 17)
        self.assertTrue(split["course_split_frozen"])
        self.assertFalse(split["original_joint_dataset_frozen"])

    def test_partition_guard_rejects_test_ids_as_training(self):
        split = inherit_split(self.rows, self.parent, "test-hash")
        with self.assertRaises(ValueError):
            check_access(self.rows, split, split["indices"]["test"][:1], "train")
        result = check_access(self.rows, split, split["indices"]["test"][:1], "test")
        self.assertTrue(result["future_confirmatory_quarantine_required"])

    def test_partition_guard_rejects_duplicate_or_forged_ids(self):
        split = inherit_split(self.rows, self.parent, "test-hash")
        row_id = split["indices"]["train"][0]
        for ids in ([row_id, row_id], ["invented"]):
            with self.assertRaises(ValueError):
                check_access(self.rows, split, ids, "train")

    def test_partial_edit_pairs_never_enter_RI_denominator(self):
        self.assertEqual(len(self.audit["RI_gap_pair_ids"]), 9)
        self.assertEqual(len(self.audit["non_RI_excluded_pair_ids"]), 3)
        self.assertTrue(all(not t["RI_gap_eligible"] for t in self.tasks if t["domain"] == "edit"))
        self.assertEqual(self.audit["mechanism_templates"], 4)
        self.assertFalse(self.audit["satisfies_60_distinct_behavior_families"])

    def test_confirmation_does_not_make_settled_side_effect_reversible(self):
        tasks = [t for t in self.tasks if t["domain"] in {"mail", "transfer"}
                 and t["label"] == "irreversible"]
        self.assertEqual(len(tasks), 6)
        for task in tasks:
            self.assertEqual(task["cautious_goal_witness"]["pre_action_labels"][-1], "irreversible")
            self.assertIsNone(task["recoverable_shortcut_goal_witness"])

    def test_file_safe_witnesses_have_reversible_commitments(self):
        for task in self.tasks:
            if task["domain"] in {"delete", "edit"}:
                witness = task["recoverable_shortcut_goal_witness"]
                self.assertIsNotNone(witness)
                self.assertEqual(witness["pre_action_labels"][-1], "reversible")
                self.assertFalse(witness["is_model_result"])

    def test_behavior_overlap_is_disclosed(self):
        self.assertEqual(self.audit["probe_pool_structural_overlap_versions"], 12)
        self.assertTrue(all("development" in t["scope"] for t in self.tasks))

    def test_invalid_requests_preserve_state_and_replay(self):
        scenario = scenario_from_task(self.tasks[0])
        episode = run_episode(scenario, lambda _: {"text": '{"tool":"shell","arguments":{}}'})
        verify_episode(episode)
        self.assertEqual(episode["status"], "consecutive_invalid_abort")
        self.assertEqual(episode["initial_hash"], episode["final_hash"])
        forged = deepcopy(episode)
        forged["trace"][0]["after_hash"] = "forged"
        with self.assertRaises(ValueError):
            verify_episode(forged)

    def test_censored_oracle_is_never_a_negative_ground_truth(self):
        scenario = scenario_from_task(next(t for t in self.tasks if t["label"] == "reversible"))
        result = label_action(scenario.state, scenario.action, max_nodes=1)
        self.assertEqual(result.label, "unknown")
        self.assertFalse(result.search_exhausted)

    def test_old_split_with_one_class_development_is_rejected(self):
        old = json.loads((ROOT / "data/splits/candidate_v5/candidate_split_1729.json").read_text())
        with self.assertRaises(ValueError):
            split_summary(self.rows, old)

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            Action.parse('{"tool":"check_state","tool":"delete_file","arguments":{}}')


if __name__ == "__main__":
    unittest.main()
