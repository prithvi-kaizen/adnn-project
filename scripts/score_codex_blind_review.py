"""Compare a sealed single-session Codex review to the frozen sandbox oracle."""

import argparse
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from course_llm_audit import (
    LABELS,
    ROOT,
    audit_input,
    batches_for,
    compare,
    load_json,
    oracle_replay,
    pack_case,
    save,
    sha,
    validate_batch,
    witness_result,
)

PACKET = ROOT / "output/llm_annotation/pilot_v1/blind_audit_A.zip"
PACKET_RECEIPT = ROOT / "docs/evidence/course_llm_annotation_v1/preparation.json"


def sealed_inputs(directory):
    design = load_json(directory / "design.json")
    seal_file = directory / "seal.json"
    seal = load_json(seal_file)
    assert seal["status"] == "ALL_100_FIRST_JUDGMENTS_SEALED_BEFORE_ORACLE_COMPARISON"
    assert not seal["scoring_undertaken"] and seal["cases"] == 100 and seal["review_passes"] == 1
    files = {"design.json"} | {f"{i:02}.{suffix}" for i in range(1, 6) for suffix in ("json", "decisions.txt")}
    assert set(seal["file_sha256"]) == files
    assert all(sha(directory / name) == fingerprint for name, fingerprint in seal["file_sha256"].items())
    assert design["human_judgments"] == 0 and design["second_reviewer"] is None
    assert not design["separate_reviewer_sessions"] and design["blinded_to_oracle_until_all_judgments_sealed"]
    assert sha(PACKET) == design["input_sha256"] == load_json(PACKET_RECEIPT)["packet_sha256"][PACKET.name]
    return design, seal_file


def score(directory):
    design, seal = sealed_inputs(directory)
    config, key, groups = audit_input()
    data = {}
    with ZipFile(PACKET) as packets:
        for i, ids in enumerate(batches_for("A", config, key), 1):
            prompt = packets.read(f"batch_{i:02}.md").decode()
            cases = json.loads(prompt.rsplit("```json\n", 1)[1].split("\n```", 1)[0])
            assert cases == [pack_case(case, key[case]) for case in ids]
            received = validate_batch(load_json(directory / f"{i:02}.json"), ids,
                                      {case: key[case]["horizon"] for case in ids})
            assert not set(data) & set(received)
            data.update(received)
    assert len(data) == len(key) == 100
    oracle_counts = oracle_replay(key)
    oracle = {case: row["label"] for case, row in key.items()}
    reviewed = {case: row["label"] for case, row in data.items()}
    witnesses = {case: witness_result(row, key[case]) for case, row in data.items()}
    correct = sum(oracle[case] == reviewed[case] for case in key)
    recovery_labels = {"reversible": "exact", "partially_reversible": "paid"}
    bad_witnesses = [case for case in sorted(key) if reviewed[case] in recovery_labels and
                     recovery_labels[reviewed[case]] != witnesses[case]]
    false_recoverable = [case for case in sorted(key) if oracle[case] == "irreversible" and
                         reviewed[case] in recovery_labels]
    disagreements = [{"case_id": case, "oracle": oracle[case], "review": reviewed[case],
                      "witness_result": witnesses[case], "reason": data[case]["reason"]}
                     for case in sorted(key) if oracle[case] != reviewed[case]]
    recalls = {label: sum(oracle[case] == reviewed[case] == label for case in key) / oracle_counts[label]
               for label in LABELS}
    rule = config["pilot_quality_rule"]
    return {"status": "COMPLETE_SINGLE_BLINDED_SANDBOX_REVIEW", "scope": design["scope"],
            "model": design["reviewer"], "same_session": True, "reviewer_count": 1, "human_judgments": 0,
            "reviewer_reviewer_agreement": None, "joint_dataset_review": False,
            "packet_sha256": sha(PACKET), "seal_sha256": sha(seal), "raw_case_count": len(data),
            "oracle_horizon_replays": 3 * len(key), "label_correct": correct, "oracle_counts": oracle_counts,
            "review_counts": dict(Counter(reviewed.values())), "recall": recalls,
            "witness_status_counts": dict(Counter(witnesses.values())), "bad_witness_cases": bad_witnesses,
            "false_recoverable_cases": false_recoverable, "disagreements": disagreements,
            "oracle_vs_review": compare(oracle, reviewed, groups),
            "meets_single_reviewer_thresholds": (correct >= rule["min_correct_of_100"] and
                min(recalls.values()) >= rule["min_class_recall"] and
                len(false_recoverable) <= rule["max_false_recoverable"] and
                len(bad_witnesses) <= rule["max_invalid_witnesses"]),
            "meets_original_two_reviewer_rule": False, "oracle_labels_modified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    save(args.output, score(args.review))
