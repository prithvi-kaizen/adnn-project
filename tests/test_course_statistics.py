import unittest

import numpy as np

from src.probes.statistics import (
    bootstrap_classification,
    execution_rates,
    group_weights,
    interval,
    paired_binary,
    weighted_classification,
)


class CourseStatisticsTests(unittest.TestCase):
    def test_no_action_is_not_an_observed_zero_action_rate(self):
        audit = {"id": "a", "pre_action_oracle_records": [], "executed_irreversible_action": False,
                 "infrastructure_failure": False, "rejected_calls": 2}
        result = execution_rates([audit], 1)
        self.assertIsNone(result["observed_execution_I_rate"])
        self.assertEqual(result["I_execution_version_incidence"], 0)
        self.assertEqual(result["rejected_calls"], 2)

    def test_unknown_or_missing_actions_yield_bounds(self):
        audit = {"id": "a", "pre_action_oracle_records": [{"oracle": {"label": "unknown"}}],
                 "executed_irreversible_action": False, "infrastructure_failure": True, "rejected_calls": 0}
        result = execution_rates([audit], 2)
        self.assertEqual(result["observed_execution_I_rate_bounds"], [0, 1])
        self.assertEqual(result["I_execution_version_incidence_bounds"], [0, 1])
        self.assertIsNone(result["I_execution_version_incidence"])
        audit["infrastructure_failure"] = False
        self.assertIsNone(execution_rates([audit], 1)["I_execution_version_incidence"])

    def test_paired_bootstrap_preserves_point_metric_and_joint_draws(self):
        record = {"ids": list("abcdef"), "truth": ["reversible", "irreversible"] * 3,
                  "predictions": ["reversible", "irreversible"] * 3,
                  "probabilities": [[.9, .1], [.1, .9]] * 3, "classes": ["reversible", "irreversible"]}
        reference = {**record, "predictions": record["predictions"][::-1],
                     "probabilities": record["probabilities"][::-1]}
        result = bootstrap_classification(record, ["a", "a", "b", "b", "c", "c"],
                                           ("reversible", "irreversible"), reference)
        self.assertEqual(result["paired_difference"]["balanced_accuracy"]["estimate"], 1)
        self.assertEqual(result["paired_difference"]["balanced_accuracy"]["CI95"], [1, 1])
        self.assertEqual(result["metrics"]["AUROC"]["defined_draws"], 1000)

    def test_missing_full_test_class_keeps_raw_BA_point_but_not_full_class_CI(self):
        record = {"ids": list("abc"), "truth": ["reversible"] * 3,
                  "predictions": ["reversible"] * 3}
        result = bootstrap_classification(record, ["a", "b", "c"], ("reversible", "irreversible"))
        self.assertEqual(result["metrics"]["balanced_accuracy"]["estimate"], 1)
        self.assertIsNone(result["metrics"]["balanced_accuracy"]["CI95"])
        self.assertEqual(result["missing_true_class_draws"], 1000)

    def test_linked_rows_receive_identical_bootstrap_multiplicity(self):
        weights, design = group_weights(["a", "a", "b", "c"], draws=20)
        np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
        np.testing.assert_array_equal(weights[:, [0, 2, 3]].sum(1), np.full(20, 3))
        self.assertEqual(design["clusters"], 3)
        np.testing.assert_array_equal(weights, group_weights(["a", "a", "b", "c"], draws=20)[0])

    def test_missing_true_class_does_not_yield_artificial_perfect_balanced_accuracy(self):
        result, complete = weighted_classification(["reversible", "irreversible"],
            ["reversible", "irreversible"], [[.9, .1], [.1, .9]],
            ["reversible", "irreversible"], ["reversible", "irreversible"], [2, 0])
        self.assertFalse(complete)
        self.assertIsNone(result["balanced_accuracy"])
        self.assertIsNone(result["AUROC"])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["macro_F1"], .5)

    def test_invalid_predictions_stay_in_accuracy_and_macro_F1_denominators(self):
        result, _ = weighted_classification(["reversible", "irreversible"], ["invalid", "irreversible"],
            None, [], ["reversible", "irreversible"], [1, 1])
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["macro_F1"], .5)

    def test_zero_discordance_has_p_one_and_zero_width_is_disclosed(self):
        result = paired_binary([1, 0, 1], [1, 0, 1], ["a", "b", "c"])
        self.assertEqual(result["exact_McNemar_p"], 1)
        self.assertEqual(result["CI95"], [0, 0])
        self.assertTrue(result["zero_width"])

    def test_too_few_clusters_cannot_produce_an_interval(self):
        self.assertIsNone(interval([.5] * 1000, 2)["CI95"])

    def test_binary_metrics_accept_real_boolean_goals_but_reject_fractions(self):
        result = paired_binary([True, False, True], [False, False, True], ["a", "b", "c"])
        self.assertEqual(result["n10"], 1)
        with self.assertRaises(ValueError):
            paired_binary([.1, 0, 1], [0, 0, 1], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
