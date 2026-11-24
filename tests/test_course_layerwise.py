import unittest

import numpy as np

from scripts.run_course_layerwise import bootstrap_design
from src.probes.layerwise import component_metrics
from src.probes.statistics import weighted_classification


class CourseLayerwiseTests(unittest.TestCase):
    def test_ties_and_missing_class_against_explicit_replicated_rows(self):
        for labels in (("irreversible", "reversible"),
                       ("reversible", "partially_reversible", "irreversible")):
            with self.subTest(labels=labels):
                classes = list(reversed(labels))
                truth = [label for label in labels for _ in range(2)]
                predictions = labels[-1:] + tuple(truth[:-1])
                probabilities = np.eye(len(labels))[np.arange(len(truth)) % len(labels)].tolist()
                record = {"truth": truth, "predictions": predictions, "classes": classes,
                          "probabilities": probabilities}
                membership = [0, 0, 1, 2][:len(truth)] + ([2, 3] if len(labels) == 3 else [])
                counts = np.array([[1, 1, 1, 1], [2, 1, 0, 3], [4, 0, 0, 0]])
                measured, complete = component_metrics(record, membership, counts, labels)
                for index, draw in enumerate(counts):
                    rows = np.repeat(np.arange(len(truth)), draw[membership])
                    expected, is_complete = weighted_classification(
                        np.array(truth)[rows], np.array(predictions)[rows], np.array(probabilities)[rows],
                        classes, labels, np.ones(len(rows)))
                    self.assertEqual(complete[index], is_complete)
                    for key, value in expected.items():
                        if value is None:
                            self.assertTrue(np.isnan(measured[key][index]))
                        else:
                            self.assertAlmostEqual(measured[key][index], value)
                self.assertFalse(complete[-1])

    def test_duplicate_draws_keep_their_original_multiplicities(self):
        groups = ["c", "b", "a", "c", "b"]
        counts, membership, inverse, _ = bootstrap_design(groups)
        draws = np.random.default_rng(9001).choice(sorted(set(groups)), size=(1000, 3))
        expected = (draws[:, :, None] == np.asarray(groups)).sum(1)
        np.testing.assert_array_equal(counts[inverse][:, membership], expected)


if __name__ == "__main__":
    unittest.main()
