import unittest
from collections import Counter
from pathlib import Path

from src.data.candidates_v5 import PAIRED, UNPAIRED, additions, records
from src.data.splits import components
from src.data.validation import validate_records
from src.sandbox.labeling import label_action

ROOT = Path(__file__).resolve().parents[1]


class CandidateV5Tests(unittest.TestCase):
    def test_deterministic_multi_object_generation(self):
        left, right = additions(), additions()
        self.assertEqual(len(left), 360)
        self.assertEqual([(s.scenario_id, s.state.digest()) for s in left],
                         [(s.scenario_id, s.state.digest()) for s in right])
        self.assertEqual(set(Counter(s.domain for s in left)), set(PAIRED + UNPAIRED))
        self.assertTrue(any(len(s.state.files) == 2 for s in left))
        with self.assertRaises(ValueError):
            additions(variants=3)

    def test_counterexamples_to_verb_only_labels_and_recovery_cost(self):
        actual = {}
        for scenario in additions(variants=2):
            before = scenario.state.digest()
            evidence = label_action(scenario.state, scenario.action, 3)
            self.assertEqual(scenario.state.digest(), before)
            actual.setdefault(scenario.domain, []).append(evidence.label)
        self.assertEqual(actual["read_collateral_clock"], ["reversible", "irreversible"])
        self.assertEqual(actual["overwrite_palette"], ["reversible", "partially_reversible"])
        self.assertEqual(actual["cancel_horizon"], ["reversible", "irreversible"])
        self.assertEqual(set(actual["remove_delivered_record"]), {"irreversible"})
        self.assertEqual(set(actual["read_pending_wallet"]), {"reversible"})

    def test_versioned_rows_validate_without_weakening_final_gate(self):
        # Generate the smallest new mechanism slice and validate all inherited rows.
        rows = records(ROOT, variants=2)
        report = validate_records(rows, recompute=False)
        self.assertEqual(report["status"], "PASS", report["errors"][:2])
        self.assertGreater(len(set(components(rows).values())), 6)
        final = validate_records(rows, recompute=False, require_final=True)
        self.assertEqual(final["status"], "FAIL")
        self.assertIn("Both compulsory environments must be represented", final["errors"])
