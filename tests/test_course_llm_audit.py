"""Synthetic return fixtures only; these do not constitute annotation evidence."""

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.course_llm_audit import (
    AUDITORS,
    audit_input,
    batches_for,
    compare,
    prepare,
    review_returns,
    save,
    validate_batch,
    witness_result,
)


class CloudAuditTests(unittest.TestCase):
    def test_missing_duplicate_and_out_of_batch_ids_are_rejected(self):
        record = {"case_id": "case_001", "label": "unknown", "reason": "Fixture.", "witness": None}
        horizons = {"case_001": 3, "case_002": 3}
        for invalid in ([], [record, record], [{**record, "case_id": "case_002"}]):
            with self.assertRaises(ValueError):
                validate_batch(invalid, ["case_001"], horizons)
        self.assertEqual(validate_batch([record], ["case_001"], horizons), {"case_001": record})

    def test_constructive_witnesses_require_exact_or_paid_state_equality(self):
        _, key, _ = audit_input()
        for label, outcome in (("reversible", "exact"), ("partially_reversible", "paid")):
            row = next(item for item in key.values()
                       if item["label"] == label and item["label_evidence"]["witness"])
            self.assertEqual(witness_result({"witness": row["label_evidence"]["witness"]}, row), outcome)
            self.assertEqual(witness_result({"witness": []}, row), "not_restored")
            self.assertEqual(witness_result({"witness": [{"tool": "snapshot", "arguments": {}}]}, row),
                             "illegal_recovery_action")

    def test_kappa_includes_unknown_and_leaves_constant_agreement_undefined(self):
        groups = {"one": "x", "two": "y", "three": "z"}
        left = {"one": "reversible", "two": "irreversible", "three": "unknown"}
        right = {**left, "three": "partially_reversible"}
        report = compare(left, right, groups)
        self.assertAlmostEqual(report["raw_agreement"]["estimate"], 2 / 3)
        self.assertAlmostEqual(report["cohens_kappa"]["estimate"], 4 / 7)
        constant = dict.fromkeys(groups, "unknown")
        self.assertIsNone(compare(constant, constant, groups)["cohens_kappa"]["estimate"])

    def test_packet_blinding_and_end_to_end_all_abstention_fixture(self):
        config, key, _ = audit_input()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packets, receipt = root / "packets", root / "receipt.json"
            prepare(packets, receipt)
            for auditor in AUDITORS:
                with ZipFile(packets / f"blind_audit_{auditor}.zip") as archive:
                    all_cases = []
                    for number, ids in enumerate(batches_for(auditor, config, key), 1):
                        text = archive.read(f"batch_{number:02}.md").decode()
                        cases = json.loads(text.split("```json\n")[1].split("\n```")[0])
                        self.assertEqual([row["case_id"] for row in cases], ids)
                        for case in cases:
                            self.assertEqual(set(case), {"case_id", "horizon", "before", "candidate"})
                            self.assertEqual(case["before"], json.loads(key[case["case_id"]]["state_text"]))
                            self.assertNotIn(key[case["case_id"]]["id"], text)
                        all_cases.extend(ids)
                    self.assertEqual(len(all_cases), 100)
                    self.assertEqual(set(all_cases), set(key))
                (root / auditor).mkdir()
                for number, ids in enumerate(batches_for(auditor, config, key), 1):
                    save(root / auditor / f"{number:02}.json", [
                        {"case_id": case, "label": "unknown", "reason": "Synthetic abstention fixture.",
                         "witness": None} for case in ids])
            metadata = {auditor: {
                "provider": "synthetic unit fixture", "model_as_displayed": auditor,
                "declared_family": auditor, "settings": "no inference", "run_date": "2000-01-01",
                "blinded_separate_sessions": True} for auditor in AUDITORS}
            (root / "metadata.json").write_text(json.dumps({"A": metadata["A"]}))
            partial = review_returns(root, packets, receipt)
            self.assertEqual(partial["status"], "SINGLE_REVIEWER_AUDITED")
            self.assertEqual(partial["complete_reviewer_returns"], ["A"])
            self.assertIsNone(partial["model_to_model"])
            (root / "metadata.json").write_text(json.dumps(metadata))
            report = review_returns(root, packets, receipt)
            self.assertFalse(report["pilot_quality_rule_met"])
            self.assertEqual(len(report["disagreements_or_invalid_witnesses"]), 100)
            self.assertEqual(report["model_to_oracle"]["A"]["correct"], 0)
            self.assertEqual(report["model_to_oracle"]["A"]["abstentions"], 100)
            self.assertIsNone(report["model_to_model"]["cohens_kappa"]["estimate"])
            self.assertEqual(report["human_annotations"], 0)
            self.assertFalse(report["original_gate_2_complete"])


if __name__ == "__main__":
    unittest.main()
