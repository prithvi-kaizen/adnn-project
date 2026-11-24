"""Full-cap software timing fixture and outcome-blind RQ4 resource choice."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.stage_course_checkpoint import file_receipt  # noqa: E402
from src.models.course_native import NativeCourseRunner, append_record  # noqa: E402
from src.models.feature_store import sha  # noqa: E402
from src.models.native_smoke import steer_last_position  # noqa: E402
from src.models.ollama_dev import api, system_memory  # noqa: E402
from src.probes.course import labels_for  # noqa: E402
from src.probes.steering import directions, resource_choice  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_site(root, rows, block):
    report = json.loads((root / "report.json").read_text())
    assert report["status"] == "PASS_INFERENCE_COMPLETE" and report["ids"] == [r["id"] for r in rows]
    parts, ids = [], []
    for shard in report["shards"]:
        path = root / shard["path"] / "features.npz"
        assert sha(path) == shard["sha256"]
        with np.load(path, allow_pickle=False) as saved:
            parts.append((saved["raw_final_last"] if block == 24 else saved["last"][block]).copy())
        ids.extend(shard["ids"])
    assert ids == report["ids"]
    vectors = np.concatenate(parts)
    assert vectors.shape == (len(rows), 896) and vectors.dtype == np.float32 and np.isfinite(vectors).all()
    return vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve timing attempts: choose a new directory")
    args.output.mkdir(parents=True)
    out = args.output
    policy_path = ROOT / "configs/course_steering_policy_v1.json"
    policy = json.loads(policy_path.read_text())
    execution = json.loads((ROOT / policy["model_execution"]).read_text())
    settings = json.loads((ROOT / execution["pilot_settings"]).read_text())
    gate = json.loads(args.gate.read_text())
    assert gate["status"] == "PASS_C4_REDUCED_SCOPE"
    started = time.monotonic()
    paths = [Path(__file__), policy_path, ROOT / "docs/COURSE_STEERING_EXECUTION.md",
             ROOT / "src/probes/steering.py", ROOT / "src/models/course_native.py", ROOT / "src/models/native_smoke.py"]
    report = {"scope": "synthetic_full_cap_timing_not_task_behavior", "started_unix": time.time(),
              "policy": policy, "execution": execution, "gate_sha256": sha(args.gate),
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths}, "requests": [],
              "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "tokenizers")}}
    save(out / "source_lock.json", report)
    try:
        assert sys.platform == "darwin" and settings["device"] == "cpu" and settings["precision"] == "float32"
        assert sha(ROOT / "src/models/course_native.py") == execution["model_runner_source_sha256"]
        data = ROOT / "data/course_fast_track/v2"
        rows = [json.loads(line) for line in (data / "sandbox_pool.jsonl").read_text().splitlines()]
        assert sha(data / "sandbox_pool.jsonl") == execution["selected_dataset_sha256"]
        probes = ROOT / "results/course_fast_track/native_mac_cpu_v1/probes"
        lock = json.loads((probes / "selection_lock.json").read_text())
        assert lock["status"] == "ALL_SELECTIONS_LOCKED_BEFORE_TEST_EVALUATION"
        selection_path = probes / (policy["direction_source_condition"] + ".json")
        selected = json.loads(selection_path.read_text())
        candidates = {int(k): v for k, v in selected["layers"].items() if int(k) > 0}
        block = min(candidates, key=lambda k: (-candidates[k]["development"]["balanced_accuracy"], k))
        assert block == selected["steering_block_candidate"]
        split = json.loads((data / "splits/pool_1800/seed_1729.json").read_text())
        indices = [i for i, row in enumerate(rows) if row["id"] in set(split["indices"]["train"])]
        vectors = read_site(ROOT / "artifacts/activations/course_native_mac_cpu_v1/features", rows, block)
        concept, metadata = directions(vectors, labels_for(rows, 3), indices)
        del vectors
        np.savez_compressed(out / "directions.npz", **concept)
        report.update(block=block, direction_metadata=metadata, training_ids=split["indices"]["train"],
                      direction_file_sha256=sha(out / "directions.npz"), selection_sha256=sha(selection_path))
        receipt = json.loads((ROOT / execution["checkpoint_receipt"]).read_text())
        snapshot = ROOT / receipt["snapshot"]
        pin = json.loads((ROOT / "docs/evidence/course_phase1_20261115/fallback_model_metadata.json").read_text())
        report["checkpoint_files"] = {f["path"]: file_receipt(snapshot / f["path"], f) for f in pin["files"]}
        if api("/api/ps", timeout=5).get("models", []):
            raise RuntimeError("Another Ollama model is resident")
        baseline = system_memory()
        report["system_memory_baseline"] = baseline

        def guard():
            measured = system_memory()
            measured.update(timestamp_unix=time.time(), free_disk_bytes=shutil.disk_usage(ROOT).free)
            measured["swap_growth_bytes"] = measured["swap_used_bytes"] - baseline["swap_used_bytes"]
            append_record(out / "resource_guards.jsonl", measured)
            if (measured["swap_growth_bytes"] > policy["guard_swap_growth_bytes"] or
                    measured["memory_free_percentage"] < policy["guard_minimum_memory_free_percentage"] or
                    measured["free_disk_bytes"] < policy["disk_reserve_bytes"] + policy["sweep_output_allowance_bytes"] or
                    time.monotonic() - started > 1200):
                raise RuntimeError("Timing resource guard failed; stop without baseline reset")

        random.seed(1729)
        np.random.seed(1729)
        torch.manual_seed(1729)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True, warn_only=True)
        runner = NativeCourseRunner(snapshot, settings, out, guard)
        assert hashlib.sha256(runner.tokenizer.chat_template.encode()).hexdigest() == execution["chat_template_sha256"]
        # Fill a synthetic fixture, without truncating any research state/history.
        def fixture(repeats):
            return [{"role": "system", "content": "This is a software runtime fixture, not an agent task."},
                    {"role": "user", "content": "Continue a sequence of words. " + "x " * repeats}]
        low, high = 0, 4096
        while low < high:
            middle = (low + high + 1) // 2
            length = len(runner.tokenizer.apply_chat_template(fixture(middle), tokenize=True, add_generation_prompt=True))
            if length <= policy["stress_prompt_target_tokens"]:
                low = middle
            else:
                high = middle - 1
        messages = fixture(low)
        _, encoded = runner.encode(messages)
        assert 1900 <= len(encoded["input_ids"]) <= 1920
        report["stress_prompt_tokens"] = len(encoded["input_ids"])
        baseline_output = runner.generate(messages, max_new_tokens=128, min_new_tokens=128)
        report["requests"].append({"condition": "ordinary", **baseline_output})
        for name, strength, key in (("concept_zero", 0, "concept"), ("concept_plus_2", 2, "concept"),
                                    ("random_811_plus_2", 2, "random_811")):
            hooks = len(runner.model.model.layers[block - 1]._forward_hooks)
            with steer_last_position(runner.model.model.layers[block - 1], torch.from_numpy(concept[key]), strength):
                response = runner.generate(messages, max_new_tokens=128, min_new_tokens=128)
            assert len(runner.model.model.layers[block - 1]._forward_hooks) == hooks
            if strength == 0:
                assert response["output_token_ids"] == baseline_output["output_token_ids"]
            report["requests"].append({"condition": name, **response})
            save(out / "progress.json", report)
        assert all(r["new_tokens"] == 128 for r in report["requests"])
        remaining = (datetime.fromisoformat(policy["operational_deadline_utc"]) - datetime.now(timezone.utc)).total_seconds()
        decision = resource_choice(max(r["seconds"] for r in report["requests"]), remaining, policy)
        report.update(status="PASS_TIMING_AND_RESOURCE_DECISION", decision=decision,
                      remaining_operational_seconds=remaining, elapsed_seconds=time.monotonic() - started,
                      memory=runner.memory(), ordinary_zero_complete_token_identity=True,
                      research_behavior_episodes=0, no_accuracy_or_response_text_used_for_scope=True)
        save(out / "report.json", report)
        print(json.dumps({"status": report["status"], "decision": decision, "block": block}, indent=2))
    except Exception:
        report.update(status="FAILED_OR_BLOCKED", error=traceback.format_exc(), elapsed_seconds=time.monotonic() - started)
        save(out / "report.json", report)
        raise


if __name__ == "__main__":
    main()
