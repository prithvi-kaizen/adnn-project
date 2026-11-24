"""Verify actual C4 records and run the full suite before passing the course gate."""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.course_metrics import audit_course_episode  # noqa: E402
from src.models.course_native import parse_stated, stated_messages  # noqa: E402
from src.models.feature_store import sha  # noqa: E402


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_generation(root, tokenizer):
    requests = read_lines(root / "generation_requests.jsonl")
    responses = read_lines(root / "generation_responses.jsonl")
    assert [r["request_id"] for r in requests] == list(range(1, len(requests) + 1))
    assert len({r["request_id"] for r in responses}) == len(responses)
    assert set(r["request_id"] for r in responses) <= set(r["request_id"] for r in requests)
    for row in responses:
        assert tokenizer.decode(row["output_token_ids"], skip_special_tokens=True) == row["text"]
        assert len(row["output_token_ids"]) == row["new_tokens"]
    for row in requests:
        rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        assert row["rendered_prompt"] == rendered and row["do_sample"] is False
        assert row["input_ids"] == tokenizer(rendered, add_special_tokens=False)["input_ids"]
        assert len(row["input_ids"]) + row["max_new_tokens"] <= 2048
    return {r["request_id"]: r for r in requests}, {r["request_id"]: r for r in responses}


def verify_stated(root, rows, tokenizer):
    report = json.loads((root / "report.json").read_text())
    assert report["status"] == "PASS_INFERENCE_COMPLETE" and report["model_calls_completed"] == len(rows)
    requests, responses = verify_generation(root, tokenizer)
    actual = read_lines(root / "stated.jsonl")
    assert len(requests) == len(responses) == len(rows)
    assert [r["id"] for r in actual] == [r["id"] for r in rows]
    for row, saved in zip(rows, actual):
        request = requests[saved["request_id"]]
        assert request["messages"] == stated_messages(row) and request["max_new_tokens"] == 64
        assert all(saved[key] == value for key, value in responses[saved["request_id"]].items())
        assert parse_stated(saved["text"]) == saved["parsed_label"]
    guards = read_lines(root / "resource_guards.jsonl")
    assert all(r["swap_growth_bytes"] <= 512 * 1024**2 and r["memory_free_percentage"] >= 10 and
               r["free_disk_bytes"] >= 2 * 1024**3 for r in guards)
    assert report["elapsed_seconds"] + report["prior_feature_seconds"] <= 21600
    return {"N": len(rows), "valid_labels": sum(r["parsed_label"] is not None for r in actual),
            "invalid_labels": sum(r["parsed_label"] is None for r in actual),
            "maximum_global_swap_growth_bytes": max(r["swap_growth_bytes"] for r in guards),
            "elapsed_seconds": report["elapsed_seconds"], "report_sha256": sha(root / "report.json")}


def verify_behavior(root, tasks, tokenizer):
    report = json.loads((root / "report.json").read_text())
    assert report["status"] == "PASS_ASSIGNED_DEVELOPMENT_RUN_COMPLETE"
    assert report["assigned_ids"] == [t["id"] for t in tasks] and report["unobserved_ids"] == []
    requests, responses = verify_generation(root, tokenizer)
    stated = read_lines(root / "stated.jsonl")
    assert [r["id"] for r in stated] == [t["id"] for t in tasks]
    assert len(report["episodes"]) == len(tasks)
    selected = {r["id"]: r for r in report["episodes"]}
    replayed, infra = 0, 0
    for task, response in zip(tasks, stated):
        assert requests[response["request_id"]]["messages"] == stated_messages(task)
        assert response["text"] == responses[response["request_id"]]["text"]
        assert parse_stated(response["text"]) == response["parsed_label"]
        # Verify originals and any whole-pair retry, not only selected episodes.
        files = [root / f"episode_{task['id']}.json"]
        if task["pair_id"] in report.get("whole_pair_infrastructure_retries", []):
            files.append(root / f"retry_episode_{task['id']}.json")
            assert selected[task["id"]]["selected_attempt"] == 2
        else:
            assert selected[task["id"]]["selected_attempt"] == 1
        for path in files:
            episode = json.loads(path.read_text())
            audit = audit_course_episode(task, episode)
            replayed += 1
            for event in episode["trace"]:
                if "generation" in event:
                    key = event["generation"]["request_id"]
                    assert event["generation"] == responses[key]
                    assert event["request_messages"] == requests[key]["messages"]
                    assert event["raw_output"] == responses[key]["text"]
            if path.name == selected[task["id"]]["episode_file"]:
                assert selected[task["id"]]["audit"] == audit
                infra += audit["infrastructure_failure"]
    return {"assigned_versions": len(tasks), "replayed_original_and_retry_episodes": replayed,
            "still_infrastructure_incomplete_versions": infra,
            "complete_behavior_outcomes": infra == 0,
            "raw_generation_requests": len(requests), "raw_generation_responses": len(responses),
            "report_sha256": sha(root / "report.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new gate receipt")
    report = {"status": "VERIFYING", "scope": "reduced_automated_sandbox_course_study", "started_unix": time.time()}
    try:
        from transformers import AutoTokenizer
        freeze = json.loads((ROOT / "docs/evidence/protocol_freeze.json").read_text())
        assert all(sha(ROOT / path) == value for path, value in freeze["sha256"].items())
        data = ROOT / "data/course_fast_track/v2"
        rows = read_lines(data / "sandbox_pool.jsonl")
        tasks = read_lines(data / "development_tasks.jsonl")
        execution = json.loads((ROOT / "configs/course_execution_native_mac_cpu_v1.json").read_text())
        assert sha(data / "sandbox_pool.jsonl") == execution["selected_dataset_sha256"]
        data_freeze = json.loads((data / "freeze_manifest.json").read_text())
        assert all(sha(data / p) == value["sha256"] for p, value in data_freeze["files"].items())
        checkpoint = json.loads((ROOT / execution["checkpoint_receipt"]).read_text())
        tokenizer = AutoTokenizer.from_pretrained(ROOT / checkpoint["snapshot"], local_files_only=True, trust_remote_code=False)
        for name, status in (("features_verification.json", "PASS_FEATURE_INTEGRITY"),
                             ("probes_verification.json", "PASS_PROBE_SELECTION_AND_METRICS"),
                             ("length_verification.json", "PASS_EXPLORATORY_LENGTH_CONTROL")):
            receipt = json.loads((args.evidence / name).read_text())
            assert receipt["status"] == status
        assert json.loads((args.evidence / "probes_verification.json").read_text())["predictions_sha256"] == sha(args.results / "probes/test_predictions.jsonl.gz")
        assert json.loads((args.evidence / "length_verification.json").read_text())["predictions_sha256"] == sha(args.results / "length/test_predictions.json")
        features_root = ROOT / "artifacts/activations/course_native_mac_cpu_v1/features"
        feature_receipt = json.loads((args.evidence / "features_verification.json").read_text())
        assert sha(features_root / "report.json") == feature_receipt["report_sha256"]
        feature_report = json.loads((features_root / "report.json").read_text())
        assert all(sha(features_root / shard["path"] / "features.npz") == shard["sha256"] for shard in feature_report["shards"])
        report["stated"] = verify_stated(args.artifacts / "stated", rows, tokenizer)
        report["behavior"] = verify_behavior(args.artifacts / "behavior", tasks, tokenizer)
        summary_path = args.results / "core_summary.json"
        summary = json.loads(summary_path.read_text())
        assert len(summary["probe_conditions"]) == 180 and len(summary["RQ2"]) == 6
        assert summary["RQ3"]["eligible_RI_pairs"] == 9 and summary["RQ3"]["eligible_template_clusters"] == 3
        assert len(summary["development_representation_knowledge_behavior"]["rows"]) == 24
        assert summary["human_judgments_created"] == 0
        for name in ("stated", "behavior"):
            assert json.loads((args.artifacts / name / "report.json").read_text())["execution"] == execution
        references = json.loads((args.artifacts / "references/report.json").read_text())
        assert references["status"] == "PASS_SCRIPTED_REFERENCE_REPLAY" and references["is_model_result"] is False
        assert references["N_reference_episodes"] == 72
        by_id = {task["id"]: task for task in tasks}
        for record in references["records"]:
            path = args.artifacts / "references" / record["file"]
            assert sha(path) == record["sha256"]
            saved = json.loads(path.read_text())
            assert saved["is_model_result"] is False
            assert audit_course_episode(by_id[record["task_id"]], saved["episode"]) == saved["audit"] == record["audit"]
        unit_log = args.evidence / "unittest_gate_c4.log"
        if unit_log.exists():
            raise ValueError("Preserve previous gate test logs")
        completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
                                   cwd=ROOT, text=True, capture_output=True)
        unit_log.write_text(completed.stdout + completed.stderr)
        assert completed.returncode == 0
        count = re.search(r"Ran (\d+) tests", completed.stderr)
        assert count
        roots = [ROOT / "artifacts/activations/course_native_mac_cpu_v1", args.results, args.artifacts, args.evidence]
        files = {p.resolve() for root in roots for p in root.rglob("*") if p.is_file()}
        files.update(p.resolve() for p in (ROOT / "artifacts/source").glob("course_core_*.tar.gz"))
        total = sum(p.stat().st_size for p in files)
        assert total <= 768 * 1024**2 and shutil.disk_usage(ROOT).free >= 2 * 1024**3
        report.update(status="PASS_C4_REDUCED_SCOPE", summary_sha256=sha(summary_path),
            source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                (Path(__file__), ROOT / "docs/COURSE_CORE_PROTOCOL.md", ROOT / "docs/COURSE_ARTIFACT_BUDGET_AMENDMENT.md")},
            test_count=int(count.group(1)), unit_log_sha256=sha(unit_log), C4_artifact_bytes=total,
            C4_artifact_allowance_bytes=768 * 1024**2,
            original_research_gates="PARTIAL_NOT_WAIVED", certified_physical_labels=0, human_annotations=0,
            prescribed_BF16_model_families_run=0, course_native_model_families_run=1,
            original_frozen_files_unchanged=True, learned_direction_RQ4="NOT_RUN_AT_C4_GATE")
    except Exception:
        report.update(status="FAILED_OR_PARTIAL", error=traceback.format_exc())
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
