"""One bounded, separately labeled exploratory first-decision intervention."""

import argparse
import importlib.metadata
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.benchmark_course_steering import read_site  # noqa: E402
from scripts.stage_course_checkpoint import file_receipt  # noqa: E402
from src.agent.course_metrics import audit_course_episode  # noqa: E402
from src.agent.loop import initial_messages, run_episode  # noqa: E402
from src.data.course import scenario_from_task  # noqa: E402
from src.models.course_native import NativeCourseRunner, append_record  # noqa: E402
from src.models.feature_store import sha  # noqa: E402
from src.models.native_smoke import steer_last_position  # noqa: E402
from src.models.ollama_dev import api, system_memory  # noqa: E402
from src.probes.course import labels_for  # noqa: E402
from src.probes.steering import directions  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve attempts: select a new output directory")
    out = args.output
    out.mkdir(parents=True)
    started = time.monotonic()
    execution = json.loads((ROOT / "configs/course_execution_native_mac_cpu_v1.json").read_text())
    settings = json.loads((ROOT / execution["pilot_settings"]).read_text())
    gate_path = ROOT / "docs/evidence/course_phase4_20261115/gate_c4.json"
    assert json.loads(gate_path.read_text())["status"] == "PASS_C4_REDUCED_SCOPE"
    data = ROOT / "data/course_fast_track/v2"
    tasks = [json.loads(line) for line in (data / "development_tasks.jsonl").read_text().splitlines()]
    tasks = [t for t in tasks if t["pair_id"] in {"pilot-delete-0", "pilot-mail-0", "pilot-transfer-0"}]
    assert len(tasks) == 6 and all(t["RI_gap_eligible"] for t in tasks)
    report = {"scope": "exploratory_first_decision_not_full_RQ4", "started_unix": time.time(), "execution": execution,
              "assigned_ids": [t["id"] for t in tasks], "full_15_turn_RQ4": "NOT_RUN",
              "human_judgments_created": 0, "cells": [], "timing_requests": [], "zero_validations": [],
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__),
                  ROOT / "docs/COURSE_FIRST_DECISION_AMENDMENT.md", ROOT / "src/probes/steering.py",
                  ROOT / "src/models/course_native.py", ROOT / "scripts/benchmark_course_steering.py")},
              "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "tokenizers", "safetensors")}}
    save(out / "source_lock.json", report)
    try:
        assert settings["device"] == "cpu" and settings["precision"] == "float32" and sys.platform == "darwin"
        assert sha(ROOT / "src/models/course_native.py") == execution["model_runner_source_sha256"]
        rows = [json.loads(line) for line in (data / "sandbox_pool.jsonl").read_text().splitlines()]
        probes = ROOT / "results/course_fast_track/native_mac_cpu_v1/probes"
        selection_path = probes / "s1729_h3_binary_last_all_ordinary.json"
        selection = json.loads(selection_path.read_text())
        block = selection["steering_block_candidate"]
        assert block == min((int(k) for k in selection["layers"] if int(k) > 0),
            key=lambda k: (-selection["layers"][str(k)]["development"]["balanced_accuracy"], k))
        split = json.loads((data / "splits/pool_1800/seed_1729.json").read_text())
        train = set(split["indices"]["train"])
        values = read_site(ROOT / "artifacts/activations/course_native_mac_cpu_v1/features", rows, block)
        vectors, metadata = directions(values, labels_for(rows, 3), [i for i, r in enumerate(rows) if r["id"] in train])
        del values
        np.savez_compressed(out / "directions.npz", **vectors)
        report.update(block=block, direction_metadata=metadata, direction_sha256=sha(out / "directions.npz"),
                      selection_sha256=sha(selection_path), episode_max_steps=1, recovery_horizon=3)
        baseline_root = ROOT / "artifacts/course_fast_track/native_mac_cpu_v1/behavior"
        baseline_report = json.loads((baseline_root / "report.json").read_text())
        assert baseline_report["execution"] == execution
        for package, version in report["packages"].items():
            assert baseline_report["packages"][package] == version
        originals = {}
        for task in tasks:
            entry = next(r for r in baseline_report["episodes"] if r["id"] == task["id"])
            assert not entry["audit"]["infrastructure_failure"]
            originals[task["id"]] = json.loads((baseline_root / entry["episode_file"]).read_text())
        receipt = json.loads((ROOT / execution["checkpoint_receipt"]).read_text())
        snapshot = ROOT / receipt["snapshot"]
        pinned = json.loads((ROOT / "docs/evidence/course_phase1_20261115/fallback_model_metadata.json").read_text())
        report["checkpoint_files"] = {f["path"]: file_receipt(snapshot / f["path"], f) for f in pinned["files"]}
        if api("/api/ps", timeout=5).get("models", []):
            raise RuntimeError("Another Ollama model is resident")
        baseline = system_memory()
        report["system_memory_baseline"] = baseline

        def guard():
            measured = system_memory()
            measured.update(timestamp_unix=time.time(), free_disk_bytes=shutil.disk_usage(ROOT).free)
            measured["swap_growth_bytes"] = measured["swap_used_bytes"] - baseline["swap_used_bytes"]
            append_record(out / "resource_guards.jsonl", measured)
            if (measured["swap_growth_bytes"] > 512*1024**2 or measured["memory_free_percentage"] < 10 or
                    measured["free_disk_bytes"] < 2*1024**3+64*1024**2 or time.monotonic()-started > 1800):
                raise RuntimeError("First-decision resource guard failed; preserve the attempt")

        random.seed(1729)
        np.random.seed(1729)
        torch.manual_seed(1729)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True, warn_only=True)
        runner = NativeCourseRunner(snapshot, settings, out, guard)
        target = runner.model.model.layers[block-1]
        captured = []
        observer = target.register_forward_hook(lambda module, inputs, output:
            captured.append((output[0] if isinstance(output, tuple) else output).detach().cpu().clone()))
        try:
            feature = runner.feature_batch([tasks[0]])
        finally:
            observer.remove()
        positions = feature["records"][0]["action_positions"]
        expected = feature["raw_final_last"][0] if block == 24 else feature["last"][block, 0]
        assert torch.equal(captured[0][0, positions[-1]], expected)
        report["exact_raw_site_binding"] = True
        del feature
        captured.clear()
        messages = [initial_messages(scenario_from_task(t)) for t in tasks]
        longest = max(messages, key=lambda m: len(runner.encode(m)[1]["input_ids"]))
        ordinary = runner.generate(longest, max_new_tokens=128, min_new_tokens=128)
        report["timing_requests"].append({"condition": "ordinary", **ordinary})
        for key, strength in (("concept", 0), ("concept", 2), ("random_811", 2)):
            with steer_last_position(target, torch.from_numpy(vectors[key]), strength):
                response = runner.generate(longest, max_new_tokens=128, min_new_tokens=128)
            if strength == 0:
                assert response["output_token_ids"] == ordinary["output_token_ids"]
            report["timing_requests"].append({"direction": key, "strength": strength, **response})
        assert all(r["new_tokens"] == 128 for r in report["timing_requests"])
        projected = 2*106*max(r["seconds"] for r in report["timing_requests"])
        report["resource_projection_seconds"] = projected
        save(out / "timing_decision.json", report)
        if projected > 1800:
            report["status"] = "NOT_RUN_PROJECTION_EXCEEDS_30_MINUTE_CAP"
            save(out / "report.json", report)
            return
        for task in tasks:
            with steer_last_position(target, torch.from_numpy(vectors["concept"]), 0):
                validation = runner.generate(initial_messages(scenario_from_task(task)))
            original = originals[task["id"]]["trace"][0]["generation"]
            assert validation["output_token_ids"] == original["output_token_ids"]
            report["zero_validations"].append({"id": task["id"], "request_id": validation["request_id"], "exact_token_identity": True})
            episode = run_episode(scenario_from_task(task), lambda messages, response=original: response, max_steps=1)
            cell = {"id": task["id"], "pair_id": task["pair_id"], "template_id": task["template_id"],
                    "initial_label": task["label"], "direction": "baseline", "strength": 0,
                    "baseline_reused": True, "episode": episode, "audit": audit_course_episode(task, episode)}
            append_record(out / "cells.jsonl", cell)
            report["cells"].append({k: cell[k] for k in ("id", "direction", "strength", "audit")})
        for key in vectors:
            for strength in (-2, -1, 1, 2):
                for task in tasks:
                    before_hooks = len(target._forward_hooks)
                    with steer_last_position(target, torch.from_numpy(vectors[key]), strength):
                        episode = run_episode(scenario_from_task(task), runner.generate, max_steps=1)
                    assert len(target._forward_hooks) == before_hooks
                    audit = audit_course_episode(task, episode)
                    cell = {"id": task["id"], "pair_id": task["pair_id"], "template_id": task["template_id"],
                            "initial_label": task["label"], "direction": key, "strength": strength,
                            "baseline_reused": False, "episode": episode, "audit": audit}
                    append_record(out / "cells.jsonl", cell)
                    report["cells"].append({k: cell[k] for k in ("id", "direction", "strength", "audit")})
                    if audit["infrastructure_failure"]:
                        raise RuntimeError("An assigned decision is infrastructure-incomplete")
                save(out / "progress.json", report)
                print(f"First-decision cells: {len(report['cells'])}/102", flush=True)
        assert len(report["cells"]) == 102 and len(report["zero_validations"]) == 6
        assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) <= 64*1024**2
        report.update(status="PASS_EXPLORATORY_FIRST_DECISION_RECORDS", elapsed_seconds=time.monotonic()-started,
                      first_decisions=102, new_generation_calls=runner.request_count, memory=runner.memory(),
                      completes_original_full_RQ4=False)
        assert runner.request_count == 106
        save(out / "report.json", report)
    except Exception:
        report.update(status="FAILED_OR_BLOCKED", error=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
        save(out / "report.json", report)
        raise


if __name__ == "__main__":
    main()
