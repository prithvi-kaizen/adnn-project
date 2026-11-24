"""Engineering tests use synthetic labels, never human annotation evidence."""

import csv
import unittest
from copy import deepcopy
from io import StringIO
from pathlib import Path

from src.data.annotation import blinded_package, cohens_kappa, completed_labels
from src.data.candidates import records
from src.data.splits import candidate_split, components, validate_split
from src.data.validation import validate_records
from src.scienceworld_adapter.replay import ExportedState, search_recovery

ROOT = Path(__file__).resolve().parents[1]


class FakeWorld:
    def __init__(self, complete=True, same=False):
        self.complete, self.same = complete, same

    def replay(self, actions):
        value = 0 if self.same else (sum(a == "forward" for a in actions)
                                    - sum(a == "back" for a in actions))
        return ExportedState(str(value), self.complete, None if self.complete else "hidden_counter")

    def legal_actions(self):
        return ("forward", "back")


class ScienceWorldSearchTests(unittest.TestCase):
    def test_complete_constructive_inverse(self):
        r = search_recovery(FakeWorld(), (), "forward")
        self.assertEqual(r["label"], "reversible")
        self.assertEqual(r["witness"], ["back"])

    def test_three_step_recovery_is_found_at_its_actual_horizon(self):
        class Chain:
            def replay(self, actions):
                self.state = 0
                for action in actions:
                    assert action == self.legal_actions()[0]
                    self.state = (self.state + 1) % 4
                return ExportedState(str(self.state), True)

            def legal_actions(self):
                return (("commit", "undo1", "undo2", "undo3")[self.state],)

        self.assertEqual(search_recovery(Chain(), (), "commit", 2)["label"], "unknown")
        result = search_recovery(Chain(), (), "commit", 3)
        self.assertEqual(result["label"], "reversible")
        self.assertEqual(result["witness"], ["undo1", "undo2", "undo3"])

    def test_incomplete_export_never_certifies_observed_recovery(self):
        r = search_recovery(FakeWorld(False, True), (), "forward", diagnostic_only=True)
        self.assertEqual(r["label"], "unknown")
        self.assertTrue(r["diagnostic_observational_matches"])
        self.assertEqual(r["duplicate_states_skipped"], 0)

    def test_incomplete_default_refuses_search(self):
        r = search_recovery(FakeWorld(False), (), "forward")
        self.assertEqual(r["expanded_nodes"], 0)
        self.assertEqual(r["censor_reason"], "hidden_counter")

    def test_budget_does_not_become_negative_label(self):
        r = search_recovery(FakeWorld(), (), "forward", max_nodes=1)
        self.assertEqual(r["label"], "unknown")
        self.assertFalse(r["search_exhausted"])
        self.assertEqual(r["censor_reason"], "search_censored")


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = records(ROOT, pairs_per_mechanism=1)

    def test_mechanism_labels_and_witnesses_recompute(self):
        report = validate_records(self.rows)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["class_counts"], {
            "reversible": 8, "partially_reversible": 6, "irreversible": 10})
        self.assertEqual(report["minimal_pair_fraction"], 1)

    def test_candidates_are_seeded_and_not_padded_pilot(self):
        self.assertEqual(self.rows, records(ROOT, pairs_per_mechanism=1))
        self.assertFalse(any(r["id"].startswith("pilot") for r in self.rows))
        self.assertTrue(all(r["scenario_id"] not in r["state_text"] + r["goal"] + r["action_text"]
                            for r in self.rows))
        self.assertFalse(validate_records(self.rows, require_final=True)["final_dataset_gate_passed"])

    def test_connected_grouping_is_more_conservative_than_named_templates(self):
        grouped = components(self.rows)
        self.assertEqual(len(set(grouped.values())), 6)
        self.assertEqual(grouped[self.rows[12]["id"]], grouped[self.rows[16]["id"]])

    def test_split_has_no_pair_template_or_near_state_leakage(self):
        result = candidate_split(self.rows, 1729)
        validate_split(self.rows, result)
        self.assertFalse(result["final_freeze"])
        bad = deepcopy(result)
        row_id = bad["indices"]["test"].pop()
        bad["indices"]["train"].append(row_id)
        with self.assertRaises(ValueError):
            validate_split(self.rows, bad)

    def test_annotation_does_not_disclose_automatic_classes_or_sibling_ids(self):
        package = blinded_package(self.rows, n=12)
        self.assertEqual(package, blinded_package(self.rows, n=12))
        for annotator in ("A", "B"):
            self.assertNotIn("automatic_label", package[f"cases_{annotator}.csv"])
            self.assertNotIn("sandbox-v4", package[f"cases_{annotator}.csv"])
            self.assertNotIn("label_evidence", package[f"cases_{annotator}.csv"])
        a = list(csv.DictReader(StringIO(package["cases_A.csv"])))
        b = list(csv.DictReader(StringIO(package["cases_B.csv"])))
        self.assertNotEqual([r["annotation_id"] for r in a], [r["annotation_id"] for r in b])
        self.assertEqual(sorted(a, key=lambda r: r["annotation_id"]), sorted(b, key=lambda r: r["annotation_id"]))
        with self.assertRaises(ValueError):
            completed_labels(package["annotator_A.csv"], {f"case_{i + 1:03d}" for i in range(12)})

    def test_synthetic_kappa_regression_and_undefined_constant_case(self):
        a = ["reversible", "reversible", "irreversible", "irreversible"]
        b = ["reversible", "irreversible", "reversible", "irreversible"]
        self.assertEqual(cohens_kappa(a, b)["cohens_kappa"], 0)
        self.assertEqual(cohens_kappa(a, a)["cohens_kappa"], 1)
        self.assertIsNone(cohens_kappa(["reversible"], ["reversible"])["cohens_kappa"])
