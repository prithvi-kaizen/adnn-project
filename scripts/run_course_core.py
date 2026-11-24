"""Sequential native core inference; immutable shards and pre-output exposure records."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
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
from transformers import AutoTokenizer  # noqa: E402

from scripts.stage_course_checkpoint import file_receipt  # noqa: E402
from src.models.course_native import (  # noqa: E402
    NativeCourseRunner,
    append_record,
    assert_context,
    feature_messages,
    log_course_access,
    parse_stated,
    stated_messages,
)
from src.models.feature_store import KEYS, FeatureChunk, sha  # noqa: E402
from src.models.ollama_dev import api, system_memory  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("features", "stated"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--features-report", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve attempts: the output directory must be new")
    args.output.mkdir(parents=True)
    out = args.output
    started = time.monotonic()
    execution = json.loads((ROOT / "configs/course_execution_native_mac_cpu_v1.json").read_text())
    settings = json.loads((ROOT / execution["pilot_settings"]).read_text())
    dataset = ROOT / execution["selected_dataset"]
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    splits = {seed: json.loads((ROOT / entry["path"]).read_text())
              for seed, entry in execution["split_manifests"].items()}
    source_paths = [Path(__file__), ROOT / "src/models/course_native.py", ROOT / "src/models/feature_store.py",
                    ROOT / "src/models/alignment.py", ROOT / "src/models/native_smoke.py",
                    ROOT / "src/data/course.py", ROOT / "src/agent/loop.py",
                    ROOT / "docs/COURSE_CORE_PROTOCOL.md", ROOT / "configs/course_execution_native_mac_cpu_v1.json",
                    ROOT / "requirements-runtime-mac-lock.txt", ROOT / execution["pilot_settings"],
                    ROOT / "src/models/ollama_dev.py"]
    report = {"scope": "reduced_automated_sandbox_course_study", "mode": args.mode,
              "started_unix": time.time(), "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "execution": execution, "python": sys.version, "platform": platform.platform(),
              "packages": {p: importlib.metadata.version(p) for p in
                           ("torch", "transformers", "numpy", "tokenizers", "safetensors", "scikit-learn")},
              "model_calls_completed": 0, "status": "PREPARING", "N": len(rows)}
    save(out / "source_lock.json", report)
    fingerprint = hashlib.sha256(json.dumps(report["source_sha256"], sort_keys=True).encode()).hexdigest()
    try:
        assert sys.platform == "darwin" and settings["device"] == "cpu" and settings["precision"] == "float32"
        assert sha(dataset) == execution["selected_dataset_sha256"]
        assert sha(ROOT / "src/models/course_native.py") == execution["model_runner_source_sha256"]
        for entry in execution["split_manifests"].values():
            assert sha(ROOT / entry["path"]) == entry["sha256"]
        frozen = json.loads((dataset.parent / "freeze_manifest.json").read_text())
        assert all(sha(dataset.parent / p) == value["sha256"] for p, value in frozen["files"].items())
        if args.mode == "stated":
            if args.selection_lock is None:
                raise ValueError("All-seed probe selection lock required before stated inference")
            locked = json.loads(args.selection_lock.read_text())
            assert locked["status"] == "ALL_SELECTIONS_LOCKED_BEFORE_TEST_EVALUATION"
            assert locked["dataset_sha256"] == sha(dataset)
            report["selection_lock"] = {"path": str(args.selection_lock), "sha256": sha(args.selection_lock)}
            if args.features_report is None:
                raise ValueError("Feature report required to enforce the combined six-hour budget")
            feature_report = json.loads(args.features_report.read_text())
            assert feature_report["status"] == "PASS_INFERENCE_COMPLETE"
            report["prior_feature_seconds"] = feature_report["elapsed_seconds"]
        receipt = json.loads((ROOT / execution["checkpoint_receipt"]).read_text())
        assert receipt["status"] == "STAGED_VERIFIED"
        snapshot = ROOT / receipt["snapshot"]
        pinned = json.loads((ROOT / "docs/evidence/course_phase1_20261115/fallback_model_metadata.json").read_text())
        report["checkpoint_files"] = {f["path"]: file_receipt(snapshot / f["path"], f) for f in pinned["files"]}
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        assert hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == execution["chat_template_sha256"]
        lengths = {}
        for name, message_fn, reserve in (("features", feature_messages, 0), ("stated", stated_messages, 64)):
            values = [len(tokenizer.apply_chat_template(message_fn(row), tokenize=True, add_generation_prompt=True))
                      for row in rows]
            assert_context(values, 2048, reserve)
            lengths[name] = {"min": min(values), "max": max(values), "reserved_generation": reserve}
        report["all_prompt_context_preflight"] = lengths
        del tokenizer
        if api("/api/ps", timeout=5).get("models", []):
            raise RuntimeError("Another Ollama model is resident; preserve unrelated work")
        baseline = system_memory()
        report["system_memory_baseline"] = baseline

        def guard():
            measured = system_memory()
            measured.update(timestamp_unix=time.time(), free_disk_bytes=shutil.disk_usage(ROOT).free)
            measured["swap_growth_bytes"] = measured["swap_used_bytes"] - baseline["swap_used_bytes"]
            append_record(out / "resource_guards.jsonl", measured)
            if (measured["swap_growth_bytes"] > 512 * 1024**2 or measured["memory_free_percentage"] < 10 or
                    measured["free_disk_bytes"] < 2 * 1024**3 or
                    time.monotonic() - started + report.get("prior_feature_seconds", 0) > 21600):
                raise RuntimeError("Resource guard failed; retain attempt without rebaselining")

        random.seed(1729)
        np.random.seed(1729)
        torch.manual_seed(1729)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True, warn_only=True)
        save(out / "preflight.json", report)
        runner = NativeCourseRunner(snapshot, settings, out, guard)
        report["load_seconds"] = runner.load_seconds
        results = []
        for start in range(0, len(rows), 128):
            batch = rows[start:start + 128]
            batch_ids = {r["id"] for r in batch}
            for split in splits.values():
                for key in ("train", "development", "test"):
                    ids = [i for i in split["indices"][key] if i in batch_ids]
                    if ids:
                        log_course_access(out / "access.jsonl", rows, split, ids, key, args.mode, execution)
            if args.mode == "features":
                chunk = FeatureChunk(out / f"shard_{start // 128:03d}", [r["id"] for r in batch], fingerprint)
            for offset, row in enumerate(batch):
                append_record(out / "case_events.jsonl", {"id": row["id"], "event": "started",
                                                          "timestamp_unix": time.time()})
                if args.mode == "features":
                    result = runner.feature_batch([row])
                    arrays = {key: result[key].numpy()[:, 0] if key in ("last", "mean") else
                              result[key].numpy()[0] for key in KEYS}
                    record = {**result["records"][0], "seconds": result["seconds"], "memory": result["memory"]}
                    with gzip.open(chunk.root / f"prompt_{offset:03d}.json.gz", "wt") as handle:
                        json.dump(record, handle, ensure_ascii=False)
                    chunk.write_row(offset, arrays)
                else:
                    result = runner.generate(stated_messages(row), max_new_tokens=64)
                    append_record(out / "stated.jsonl", {"id": row["id"], "parsed_label": parse_stated(result["text"]),
                                                        **result})
                report["model_calls_completed"] += 1
                append_record(out / "case_events.jsonl", {"id": row["id"], "event": "completed",
                                                          "timestamp_unix": time.time()})
            if args.mode == "features":
                complete = chunk.finalize()
                results.append({"path": str(chunk.root.relative_to(out)), **complete})
            used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
            if args.mode == "stated":
                used += feature_report["files_bytes"]
            if used > 384 * 1024**2:
                raise RuntimeError("Combined inference artifact allowance exceeded")
            save(out / "progress.json", {"completed": report["model_calls_completed"], "N": len(rows),
                                         "elapsed_seconds": time.monotonic() - started})
            print(f"{args.mode}: {report['model_calls_completed']}/{len(rows)} complete", flush=True)
        report.update(status="PASS_INFERENCE_COMPLETE", elapsed_seconds=time.monotonic() - started,
                      ids=[r["id"] for r in rows], shards=results, memory=runner.memory())
        report["files_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        save(out / "report.json", report)
    except Exception:
        report.update(status="FAILED_OR_BLOCKED", elapsed_seconds=time.monotonic() - started,
                      error=traceback.format_exc())
        save(out / "report.json", report)
        raise


if __name__ == "__main__":
    main()
