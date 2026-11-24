import unittest

from scripts.verify_course_core import independently_measure
from src.probes.course import condition_id, conditions, eligible_indices, partitions, verb_labels


def row(name, pair, label, tool="delete_file"):
    return {"id": name, "minimal_pair_id": pair, "labels_by_horizon": {"3": {"label": label}},
            "action_text": '{"tool":"' + tool + '","arguments":{}}'}


class CourseProbeTests(unittest.TestCase):
    def test_independent_verifier_rejects_an_invented_score(self):
        record = {"ids": ["a", "b"], "truth": ["reversible", "irreversible"],
                  "predictions": ["reversible", "irreversible"],
                  "probabilities": [[.9, .1], [.1, .9]], "classes": ["reversible", "irreversible"],
                  "metrics": {"N": 2, "accuracy": 1, "balanced_accuracy": 1, "macro_F1": 1, "AUROC": 1}}
        independently_measure(record, ("reversible", "irreversible"))
        record["metrics"]["accuracy"] = .5
        with self.assertRaises(AssertionError):
            independently_measure(record, ("reversible", "irreversible"))

    def test_binary_minimal_pairs_never_keep_an_orphan_R_endpoint(self):
        rows = [row("a", "rp", "reversible"), row("b", "rp", "partially_reversible"),
                row("c", "ri", "reversible"), row("d", "ri", "irreversible"),
                row("e", None, "irreversible")]
        self.assertEqual(eligible_indices(rows, 3, True, True).tolist(), [2, 3])
        self.assertEqual(eligible_indices(rows, 3, False, True).tolist(), [0, 1, 2, 3])

    def test_partition_membership_is_id_based(self):
        rows = [row("a", None, "reversible"), row("b", None, "irreversible")]
        result = partitions(rows, {"indices": {"train": ["b"], "test": ["a"]}},
                            dict(horizon=3, binary=True, subset="all"))
        self.assertEqual(result["train"].tolist(), [1])
        self.assertEqual(result["test"].tolist(), [0])

    def test_verbs_have_fixed_targets_across_order_and_partition(self):
        rows = [row("a", None, "reversible"), row("b", None, "irreversible", "send_mail")]
        y, mapping = verb_labels(rows, ("reversible", "irreversible"))
        reversed_y, reversed_mapping = verb_labels(rows[::-1], ("reversible", "irreversible"))
        self.assertEqual(mapping, reversed_mapping)
        self.assertEqual(y.tolist(), reversed_y[::-1].tolist())

    def test_complete_condition_list_is_unique_and_prespecified(self):
        values = conditions()
        self.assertEqual(len(values), 180)
        self.assertEqual(len({condition_id(c) for c in values}), 180)
        self.assertEqual({c["seed"] for c in values}, {1729, 2718, 31415})


if __name__ == "__main__":
    unittest.main()
