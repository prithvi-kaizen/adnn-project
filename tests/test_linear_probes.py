import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.probes.linear import choose_layer, choose_linear, metrics, training_majority


class LinearProbeTests(unittest.TestCase):
    def test_iteration_limit_cannot_be_reported_as_a_converged_fit(self):
        original = LogisticRegression.fit

        def capped(estimator, x, y):
            original(estimator, x, y)
            estimator.n_iter_ = np.array([5000])
            return estimator

        with patch.object(LogisticRegression, "fit", capped), self.assertRaises(RuntimeError):
            choose_linear([[0], [1]], ["reversible", "irreversible"],
                          [[0], [1]], ["reversible", "irreversible"],
                          label_order=("reversible", "irreversible"))

    def test_scaling_is_fit_on_training_only(self):
        x = np.array([[0.], [1.], [2.], [3.]], dtype=np.float32)
        y = ["reversible", "reversible", "irreversible", "irreversible"]
        pipeline, choice = choose_linear(x, y, x + 100, y,
                                         label_order=("reversible", "irreversible"))
        self.assertEqual(pipeline.named_steps["standardscaler"].mean_.tolist(), [1.5])
        self.assertFalse(choice["test_data_used_for_selection"])
        self.assertEqual(choice["selected_C"], .01)

    def test_layer_ties_choose_shallower_layer(self):
        choices = {2: {"development": {"balanced_accuracy": .7}},
                   0: {"development": {"balanced_accuracy": .7}},
                   1: {"development": {"balanced_accuracy": .6}}}
        self.assertEqual(choose_layer(choices), 0)

    def test_binary_AUROC_has_irreversible_positive(self):
        r = metrics(["reversible", "irreversible"], ["reversible", "irreversible"],
                    [[.1, .9], [.9, .1]], ["irreversible", "reversible"],
                    label_order=("reversible", "irreversible"))
        self.assertEqual(r["AUROC"], 1)

    def test_missing_class_AUROC_is_undefined_not_zero(self):
        r = metrics(["reversible"], ["reversible"], [[.1, .9]],
                    ["irreversible", "reversible"], label_order=("reversible", "irreversible"))
        self.assertIsNone(r["AUROC"])
        self.assertEqual(r["AUROC_undefined_reason"], "missing_true_or_trained_class")

    def test_one_class_logistic_control_is_not_fabricated(self):
        with self.assertRaises(ValueError):
            choose_linear([[0], [1]], ["reversible"] * 2, [[2]], ["reversible"])

    def test_majority_ties_use_the_registered_label_order(self):
        self.assertEqual(training_majority(["irreversible", "reversible"]), "reversible")


if __name__ == "__main__":
    unittest.main()
