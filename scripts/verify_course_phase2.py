"""Verify immutable course data/tasks and their separate acceptance evidence."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.course import (  # noqa: E402
    canonical,
    development_tasks,
    select_nested_subset,
    split_summary,
)
from src.data.pilot import simulator_source_hash  # noqa: E402
from src.data.validation import validate_records  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(data_dir: Path) -> dict:
    frozen = read(data_dir / "freeze_manifest.json")
    assert frozen["course_data_task_freeze"] and not frozen["original_joint_dataset_freeze"]
    for name, entry in frozen["files"].items():
        path = data_dir / name
        assert path.resolve().is_relative_to(data_dir.resolve())
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], name
    for name, expected in frozen["original_protocol_hashes"].items():
        assert sha(ROOT / name) == expected, name
    report = read(data_dir / "validation.json")
    for name, expected in report["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    payload = data_dir / "sandbox_pool.jsonl"
    rows = [json.loads(line) for line in payload.read_text().splitlines()]
    assert len(rows) == len({r["id"] for r in rows}) == 1800
    assert all(r["scope"] == "course_automatic_sandbox_v2" and
               not r["original_joint_dataset_eligible"] for r in rows)
    qa = validate_records(rows, recompute=False, expected_source_hash=simulator_source_hash(ROOT))
    assert qa["status"] == "PASS" and qa["verified_minimal_pair_members"] == 1620
    assert report["seeded_full_regeneration"] == "PASS_1800_records_all_fields"
    subset = select_nested_subset(rows)
    assert [r["id"] for r in subset] == read(data_dir / "subset_360.json")["ids"]
    assert len(subset) == 360
    split_summary_report = {}
    for pool, selected in (("pool_1800", rows), ("subset_360", subset)):
        tests, allocations, summaries = [], [], {}
        for seed in (1729, 2718, 31415):
            split = read(data_dir / f"splits/{pool}/seed_{seed}.json")
            assert split["dataset_sha256"] == sha(payload)
            assert split["selected_ids_sha256"] == hashlib.sha256(canonical(
                sorted(r["id"] for r in selected)).encode()).hexdigest()
            assert split["course_split_frozen"] and not split["original_joint_dataset_frozen"]
            summary = split_summary(selected, split)
            assert summary == split["summary"]
            assert split["counts"]["test"] * 5 == len(selected)
            assert all(v["components"] >= 3 for v in summary.values())
            allocation = {i: s for s, ids in split["indices"].items() for i in ids}
            assert allocation not in allocations
            allocations.append(allocation)
            tests.append(set(split["indices"]["test"]))
            summaries[str(seed)] = {s: {"N": v["rows"], "components": v["components"]}
                                   for s, v in summary.items()}
        assert len({tuple(sorted(ids)) for ids in tests}) == 3
        split_summary_report[pool] = {
            "partitions": summaries,
            "test_overlap_counts": {f"{a}-{b}": len(tests[a] & tests[b])
                                    for a, b in ((0, 1), (0, 2), (1, 2))},
            "overlapping_seeds_are_not_independent_N": True,
        }
    tasks = [json.loads(line) for line in (data_dir / "development_tasks.jsonl").read_text().splitlines()]
    regenerated_tasks, task_audit = development_tasks(rows)
    assert canonical(tasks) == canonical(regenerated_tasks)
    assert task_audit == read(data_dir / "behavior_audit.json")
    assert len(task_audit["RI_gap_pair_ids"]) == 9 and len(task_audit["non_RI_excluded_pair_ids"]) == 3
    assert task_audit["probe_pool_structural_overlap_versions"] == 12
    assert len(task_audit["no_certified_recoverable_shortcut_goal_witness_ids"]) == 6
    access = read(data_dir / "access_policy.json")
    assert access["model_result_exposures_at_freeze"] == 0
    assert access["require_choice_record_before_test_generation_or_probe_evaluation"]
    assert report["certified_physical_labels"] == report["human_judgments"] == report["model_calls"] == 0
    receipt_path = ROOT / "docs/evidence/course_phase2_20261115/unittest_receipt.json"
    tests = read(receipt_path)
    assert tests["status"] == "PASS" and tests["tests"] == 114 and tests["skipped"] == 0
    assert sha(ROOT / tests["log"]) == tests["log_sha256"]
    for path, expected in tests["source_sha256"].items():
        assert sha(ROOT / path) == expected, path
    return {
        "course_gate_C2": "PASS_REDUCED_AUTOMATIC_SANDBOX_SCOPE", "original_Gate2": "PARTIAL",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest_sha256": sha(data_dir / "freeze_manifest.json"),
        "pool_N": 1800, "subset_N": 360, "minimal_pair_fraction": .9,
        "split_verification": split_summary_report, "unit_tests": 114,
        "behavior_pairs": 12, "RI_diagnostic_pairs": 9, "mechanism_templates": 4,
        "behavior_scope": "previously_exposed_development_not_independent_heldout_families",
        "certified_physical_labels": 0, "human_judgments": 0,
        "new_model_calls": 0, "weight_downloads": 0,
        "remaining_original_gaps": ["certified_ScienceWorld_and_transfer", "at_least_2000_joint_examples",
                                    "100_actual_double_annotations", "60_independent_paired_behavior_families"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/course_fast_track/v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve old receipts; choose a new output path")
    result = verify(args.data_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS C2 for the reduced course scope: immutable data/tasks, grouped splits, QA and 114 tests")
    print("Original Gate 2 PARTIAL. No new pretrained inference, physical labels or human judgments.")


if __name__ == "__main__":
    main()
