import unittest

import numpy as np

from src.probes.steering import directions, raw_site_vectors, resource_choice


class CourseSteeringTests(unittest.TestCase):
    def test_final_site_uses_raw_output_instead_of_normalized_layer(self):
        features = {"last": np.full((25, 2, 3), 99, np.float32),
                    "raw_final_last": np.full((2, 3), 7, np.float32)}
        np.testing.assert_array_equal(raw_site_vectors(features, 24), features["raw_final_last"])
        with self.assertRaises(ValueError):
            raw_site_vectors(features, 0)

    def test_direction_excludes_nontraining_and_partial_rows(self):
        values = np.array([[0, 0], [2, 0], [100, 100], [-100, 300]], dtype=np.float32)
        labels = ["reversible", "irreversible", "partially_reversible", "irreversible"]
        result, metadata = directions(values, labels, [0, 1, 2])
        np.testing.assert_array_equal(result["concept"], [2, 0])
        self.assertEqual(metadata["partial_training_examples_ignored"], 1)
        for value in result.values():
            self.assertAlmostEqual(float(np.linalg.norm(value)), 2, places=6)
        repeated, _ = directions(values, labels, [0, 1, 2])
        np.testing.assert_array_equal(result["random_811"], repeated["random_811"])

    def test_zero_direction_is_not_replaced_by_an_invented_signal(self):
        with self.assertRaises(ValueError):
            directions(np.ones((2, 3)), ["reversible", "irreversible"], [0, 1])

    def test_resource_choice_preserves_controls_and_delivery_reserve(self):
        policy = {"maximum_sweep_seconds": 28800, "delivery_reserve_seconds": 36000, "runtime_multiplier": 2}
        result = resource_choice(8, 100000, policy)
        self.assertEqual(result["selected"]["versions"], 6)
        self.assertFalse(result["selection_uses_accuracy_caution_goal_or_output_text"])
        self.assertIsNone(resource_choice(8, 36000, policy)["selected"])


if __name__ == "__main__":
    unittest.main()
