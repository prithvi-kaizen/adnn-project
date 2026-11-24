"""Freeze an automatic sandbox course pool, never the original joint study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_causal_checkpoint import verify as verify_physical_evidence  # noqa: E402
from src.agent.loop import PROMPT_VERSION, SYSTEM  # noqa: E402
from src.data.annotation import validate_blank_package  # noqa: E402
from src.data.candidates_v5 import records as regenerate  # noqa: E402
from src.data.course import (  # noqa: E402
    allocate_course_split,
    canonical,
    development_tasks,
    inherit_split,
    select_nested_subset,
)
from src.data.pilot import simulator_source_hash  # noqa: E402
from src.data.validation import validate_records  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(canonical(row) + "\n" for row in rows))


def freeze(output: Path) -> dict:
    if output.exists():
        raise ValueError("Preserve the existing freeze; choose a new version")
    source = ROOT / "data/candidates/v5/sandbox_pairs.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    source_hash = simulator_source_hash(ROOT)
    validation = validate_records(rows, expected_source_hash=source_hash)
    if validation["status"] != "PASS":
        raise ValueError(validation["errors"])
    regenerated = regenerate(ROOT)
    if canonical(regenerated) != canonical(rows):
        raise ValueError("Seeded regeneration differs from the saved candidates")
    physical = verify_physical_evidence()
    package = {p.name: p.read_text() for p in (ROOT / "data/annotation/preview_v5").glob("*.csv")}
    annotation = validate_blank_package(package)
    if annotation["status"] != "PASS":
        raise ValueError("Existing blank annotation forms do not validate")
    # Scope changes are explicit; the original candidate records remain untouched.
    course_rows = [dict(row, parent_scope=row["scope"], scope="course_automatic_sandbox_v2",
                        original_joint_dataset_eligible=False) for row in rows]
    subset = select_nested_subset(course_rows)
    tasks, task_audit = development_tasks(course_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="course-freeze-", dir=output.parent) as temporary:
        staging = Path(temporary) / "v2"
        staging.mkdir()
        write_rows(staging / "sandbox_pool.jsonl", course_rows)
        subset_ids = [row["id"] for row in subset]
        write_json(staging / "subset_360.json", {
            "selection_seed": 4601, "ids": subset_ids, "rows": len(subset),
            "parent_dataset_sha256": sha(staging / "sandbox_pool.jsonl"),
            "selection_rule": "one_fifth_of_sorted_pair_or_single_units_per_mechanism_no_labels",
            "selection_precedes_new_model_test_outcomes": True,
            "subset_is_separate_jsonl_copy": False,
        })
        splits = {}
        allocations = []
        for seed in (1729, 2718, 31415):
            allocations.append(allocate_course_split(course_rows, subset, seed, allocations))
        for name, selected in (("pool_1800", course_rows), ("subset_360", subset)):
            splits[name] = {}
            for parent in allocations:
                seed = parent["seed"]
                parent_path = ROOT / f"data/splits/candidate_v5/candidate_split_{seed}.json"
                historical = json.loads(parent_path.read_text())
                if historical["dataset_sha256"] != sha(source):
                    raise ValueError("Parent split is for a different dataset")
                manifest = inherit_split(selected, parent, sha(staging / "sandbox_pool.jsonl"))
                manifest.update(historical_provisional_split_sha256=sha(parent_path),
                                allocation_provenance={k: parent[k] for k in (
                                    "allocator", "scipy_version", "objective", "coverage_constraint",
                                    "different_from_previous_seed_allocations")},
                                selected_ids_sha256=hashlib.sha256(canonical(
                                    sorted(r["id"] for r in selected)).encode()).hexdigest())
                write_json(staging / "splits" / name / f"seed_{seed}.json", manifest)
                splits[name][str(seed)] = {key: manifest[key] for key in
                                          ("counts", "fractions", "component_count", "summary")}
        write_rows(staging / "development_tasks.jsonl", tasks)
        write_json(staging / "behavior_audit.json", task_audit)
        write_json(staging / "access_policy.json", {
            "status": "LOCKED_BEFORE_NEW_MODEL_OUTCOMES", "model_result_exposures_at_freeze": 0,
            "chosen_inference_pool": "PENDING_EXCLUDED_PILOT_RESOURCE_PROJECTION_ONLY",
            "allowed_choices": ["pool_1800", "subset_360"],
            "require_choice_record_before_test_generation_or_probe_evaluation": True,
            "selection_may_use": ["excluded_pilot_latency", "peak_memory", "artifact_bytes", "remaining_time"],
            "selection_must_not_use": ["course_test_predictions", "test_accuracy", "favorable_behavior"],
            "train_standardization_and_direction_only": True, "development_C_and_layer_selection_only": True,
            "check_access_function": "src.data.course.check_access",
            "future_runner_must_log_ids_components_seed_model_and_stage_before_output": True,
            "test_components_exposed_by_course_require_future_confirmatory_quarantine": True,
            "three_seeds_overlap_is_not_independent_replication": True,
            "do_not_adapt_protocol_after_first_seed_test_outcomes": True,
            "behavior_suite": "already_exposed_development_diagnostic_only",
            "behavior_RI_denominator_pair_ids": task_audit["RI_gap_pair_ids"],
        })
        (staging / "agent_system_prompt.txt").write_text(SYSTEM + "\n")
        write_json(staging / "prompt_lock.json", {
            "agent_prompt_version": PROMPT_VERSION,
            "agent_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "prompt_revision_requires_new_lock_before_test_outputs": True,
            "native_feature_and_stated_prompts": "lock_in_C3_after_excluded_pilot_before_holdout_use",
            "no_model_calls_in_this_freeze": True,
        })
        subset_qa = validate_records(subset, expected_source_hash=source_hash)
        if subset_qa["status"] != "PASS":
            raise ValueError(subset_qa["errors"])
        report = {
            "course_gate_C2": "PASS_REDUCED_AUTOMATIC_SANDBOX_SCOPE",
            "original_Gate2": "PARTIAL", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "worktree_dirty_at_generation": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True)),
            "generator": "scripts/freeze_course_phase2.py", "python": sys.version,
            "platform": platform.platform(), "simulator_source_sha256": source_hash,
            "candidate_source": str(source.relative_to(ROOT)), "candidate_source_sha256": sha(source),
            "seeded_full_regeneration": "PASS_1800_records_all_fields",
            "revalidation": validation, "subset_revalidation": subset_qa, "splits": splits,
            "by_horizon": {str(h): dict(Counter(r["labels_by_horizon"][str(h)]["label"] for r in rows))
                           for h in (1, 2, 3)},
            "behavior": task_audit, "certified_physical_labels": 0,
            "physical_checkpoint": {k: physical[k] for k in (
                "engine_adversarial_fixtures", "replay_prefix_settings", "guarded_searches",
                "search_unknown_counts", "instructor_approval", "methodology_amendment_applied")},
            "new_ScienceWorld_runtime_calls": 0, "physical_status": "UNKNOWN_EXCLUDED",
            "annotation_forms": annotation, "human_judgments": 0,
            "human_agreement": "NOT_PERFORMED", "model_calls": 0, "weight_downloads": 0,
            "subset_selection_uses_model_outcomes": False,
            "provenance_scope": "single_machine_fresh_regeneration_prior_cross_machine_receipts_only",
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                Path(__file__).resolve(), ROOT / "src/data/course.py", ROOT / "src/agent/loop.py",
                ROOT / "src/data/candidates_v5.py", ROOT / "src/data/candidates.py",
                ROOT / "src/data/validation.py", ROOT / "src/data/splits.py",
                ROOT / "tests/test_course_data.py", ROOT / "configs/course_fast_track.json")},
        }
        write_json(staging / "validation.json", report)
        files = {str(p.relative_to(staging)): {"sha256": sha(p), "bytes": p.stat().st_size}
                 for p in staging.rglob("*") if p.is_file()}
        write_json(staging / "freeze_manifest.json", {
            "version": "course-sandbox-v2", "course_data_task_freeze": True,
            "original_joint_dataset_freeze": False, "files": files,
            "original_protocol_hashes": json.loads((ROOT / "docs/evidence/protocol_freeze.json").read_text())["sha256"],
            "change_policy": "new_version_required_never_overwrite_this_freeze",
        })
        os.rename(staging, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/course_fast_track/v2")
    args = parser.parse_args()
    result = freeze(args.output_dir)
    print(json.dumps({key: result[key] for key in (
        "course_gate_C2", "original_Gate2", "seeded_full_regeneration", "certified_physical_labels",
        "human_judgments", "model_calls", "weight_downloads")}, indent=2))
    print("Behavior: 12 development pairs; 9 R/I eligible, 3 R/P excluded; four mechanism templates.")


if __name__ == "__main__":
    main()
