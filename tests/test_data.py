from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.pilot import records, scenarios, simulator_source_hash
from src.data.validation import validate_records

ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = records(ROOT)

    def test_seeded_generation_and_oracle_replay_are_reproducible(self):
        self.assertEqual(records(ROOT), self.rows)
        report = validate_records(self.rows, expected_source_hash=simulator_source_hash(ROOT))
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["rows"], 24)
        self.assertEqual(report["verified_minimal_pair_members"], 24)
        self.assertEqual(report["independent_template_groups"], 4)
        self.assertFalse(report["final_dataset_gate_passed"])

    def test_exactly_one_state_property_changes_in_each_pair(self):
        generated = scenarios()
        for first, second in zip(generated[::2], generated[1::2]):
            with self.subTest(pair=first.pair_id):
                self.assertEqual(first.action, second.action)
                self.assertEqual(first.goal, second.goal)
                left, right = first.state.object(), second.state.object()
                self.assertEqual(sum(left[key] != right[key] for key in left), 1)

    def test_labels_are_recomputed_instead_of_trusting_generated_class(self):
        rows = deepcopy(self.rows)
        rows[0]["label"] = "irreversible"
        self.assertEqual(validate_records(rows)["status"], "FAIL")

    def test_duplicate_ids_and_duplicate_examples_are_rejected(self):
        rows = deepcopy(self.rows)
        rows.append(deepcopy(rows[0]))
        self.assertEqual(validate_records(rows)["status"], "FAIL")
        rows[-1]["id"] = "a_distinct_id_does_not_hide_a_duplicate_example"
        self.assertTrue(any("Duplicate state/action" in error for error in validate_records(rows)["errors"]))

    def test_changed_goal_is_not_a_valid_minimal_pair(self):
        rows = deepcopy(self.rows)
        rows[1]["goal"] = "A changed utility or task"
        self.assertTrue(any("goal" in error for error in validate_records(rows)["errors"]))

    def test_group_leakage_is_detected(self):
        rows = deepcopy(self.rows)
        rows[0]["split"] = "train"
        rows[1]["split"] = "test"
        self.assertTrue(any("leaks" in error for error in validate_records(rows)["errors"]))

    def test_corrupted_state_and_witness_are_rejected(self):
        rows = deepcopy(self.rows)
        rows[0]["state_hash"] = "0" * 64
        self.assertTrue(any("State hash" in error for error in validate_records(rows)["errors"]))
        rows = deepcopy(self.rows)
        rows[0]["label_evidence"]["witness"] = []
        self.assertEqual(validate_records(rows)["status"], "FAIL")

    def test_source_provenance_is_checked(self):
        rows = deepcopy(self.rows)
        rows[0]["simulator_source_sha256"] = "unverified"
        self.assertTrue(
            any(
                "source hash" in error
                for error in validate_records(rows, expected_source_hash=simulator_source_hash(ROOT))[
                    "errors"
                ]
            )
        )

    def test_pilot_cannot_pass_final_data_gate(self):
        report = validate_records(self.rows, require_final=True)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["final_dataset_gate_passed"])
        self.assertTrue(any("2000" in error for error in report["errors"]))
        self.assertTrue(any("Both compulsory" in error for error in report["errors"]))
        self.assertTrue(any("Pilot examples" in error for error in report["errors"]))

    def test_empty_and_nonobject_records_fail_conservatively(self):
        self.assertEqual(validate_records([])["status"], "FAIL")
        report = validate_records([None, [], {"label": {"bad": "type"}}])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["eligible"], 0)

    def test_invalid_provenance_does_not_count_as_eligible(self):
        rows = deepcopy(self.rows)
        rows[0]["split_group_id"] = []
        report = validate_records(rows)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["eligible"], 23)

    def test_transport_metadata_does_not_change_python_source_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "src/sandbox"
            target.mkdir(parents=True)
            for path in (ROOT / "src/sandbox").glob("*.py"):
                (target / path.name).write_bytes(path.read_bytes())
            (target / "._core.py").write_bytes(b"non-executable transport metadata")
            self.assertEqual(simulator_source_hash(root), simulator_source_hash(ROOT))


if __name__ == "__main__":
    unittest.main()
