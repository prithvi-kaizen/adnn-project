"""Sequential fixed development behavior, with raw output and independent replay."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
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

from scripts.run_course_native_pilot import pilot_exposure  # noqa: E402
from scripts.stage_course_checkpoint import file_receipt  # noqa: E402
from src.agent.course_metrics import audit_course_episode  # noqa: E402
from src.agent.loop import run_episode, verify_episode  # noqa: E402
from src.data.course import scenario_from_task  # noqa: E402
from src.models.course_native import (  # noqa: E402
    NativeCourseRunner,
    append_record,
    parse_stated,
    stated_messages,
)
from src.models.feature_store import sha  # noqa: E402
from src.models.ollama_dev import api, system_memory  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve attempts: choose a new output directory")
    out = args.output
    out.mkdir(parents=True)
    started = time.monotonic()
    execution = json.loads((ROOT / "configs/course_execution_native_mac_cpu_v1.json").read_text())
    settings = json.loads((ROOT / execution["pilot_settings"]).read_text())
    data = ROOT / "data/course_fast_track/v2"
    tasks = [json.loads(line) for line in (data / "development_tasks.jsonl").read_text().splitlines()]
    paths = [Path(__file__), ROOT / "src/models/course_native.py", ROOT / "src/agent/course_metrics.py",
             ROOT / "src/agent/loop.py", ROOT / "scripts/run_course_native_pilot.py",
             ROOT / "docs/COURSE_BEHAVIOR_EXECUTION.md", data / "development_tasks.jsonl"]
    report = {"scope": "previously_exposed_development_diagnostic_not_60_family_study",
              "execution": execution, "started_unix": time.time(), "assigned_ids": [t["id"] for t in tasks],
              "python": sys.version, "platform": platform.platform(),
              "packages": {p: importlib.metadata.version(p) for p in
                           ("torch", "transformers", "tokenizers", "numpy", "safetensors")},
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
              "episodes": [], "unobserved_ids": [t["id"] for t in tasks], "status": "PREPARING"}
    save(out / "source_lock.json", report)
    try:
        assert sys.platform == "darwin" and settings["device"] == "cpu"
        assert sha(ROOT / "src/models/course_native.py") == execution["model_runner_source_sha256"]
        frozen = json.loads((data / "freeze_manifest.json").read_text())
        assert all(sha(data / p) == value["sha256"] for p, value in frozen["files"].items())
        receipt = json.loads((ROOT / execution["checkpoint_receipt"]).read_text())
        snapshot = ROOT / receipt["snapshot"]
        pinned = json.loads((ROOT / "docs/evidence/course_phase1_20261115/fallback_model_metadata.json").read_text())
        report["checkpoint_files"] = {f["path"]: file_receipt(snapshot / f["path"], f) for f in pinned["files"]}
        if api("/api/ps", timeout=5).get("models", []):
            raise RuntimeError("Another Ollama model is resident")
        baseline = system_memory()
        report["system_memory_baseline"] = baseline
        globally_stopped = False

        def guard():
            nonlocal globally_stopped
            measured = system_memory()
            measured.update(timestamp_unix=time.time(), free_disk_bytes=shutil.disk_usage(ROOT).free)
            measured["swap_growth_bytes"] = measured["swap_used_bytes"] - baseline["swap_used_bytes"]
            append_record(out / "resource_guards.jsonl", measured)
            if (measured["swap_growth_bytes"] > 512 * 1024**2 or measured["memory_free_percentage"] < 10 or
                    measured["free_disk_bytes"] < 2 * 1024**3 or time.monotonic() - started > 10800):
                globally_stopped = True
                raise RuntimeError("Global resource guard failed; no automatic baseline reset")

        random.seed(1729)
        np.random.seed(1729)
        torch.manual_seed(1729)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True, warn_only=True)
        for task in tasks:
            pilot_exposure(out, task, data, execution)
        runner = NativeCourseRunner(snapshot, settings, out, guard)
        assert hashlib.sha256(runner.tokenizer.chat_template.encode()).hexdigest() == execution["chat_template_sha256"]
        for task in tasks:
            response = runner.generate(stated_messages(task), max_new_tokens=64)
            append_record(out / "stated.jsonl", {"id": task["id"], "parsed_label": parse_stated(response["text"]),
                                                **response})
        for task in tasks:
            episode = run_episode(scenario_from_task(task), runner.generate, max_steps=15,
                record_event=lambda event: append_record(out / "agent_events.jsonl", {"id": task["id"], **event}))
            verify_episode(episode)
            save(out / f"episode_{task['id']}.json", episode)
            audit = audit_course_episode(task, episode)
            save(out / f"audit_{task['id']}.json", audit)
            report["episodes"].append({"id": task["id"], "status": episode["status"], "steps": episode["steps"],
                                       "audit": audit, "replay_verified": True,
                                       "episode_file": f"episode_{task['id']}.json", "selected_attempt": 1})
            report["unobserved_ids"].remove(task["id"])
            save(out / "progress.json", report)
            print(f"behavior: {len(report['episodes'])}/24 replayed", flush=True)
            if globally_stopped:
                raise RuntimeError("Stop assignments after the global guard failure")
        # One identical whole-pair retry for infrastructure failures only. The
        # original episodes stay intact; never select the more favorable version.
        by_id = {t["id"]: t for t in tasks}
        retry_pairs = {by_id[r["id"]]["pair_id"] for r in report["episodes"]
                       if r["audit"]["infrastructure_failure"]}
        report["whole_pair_infrastructure_retries"] = sorted(retry_pairs)
        for pair in sorted(retry_pairs):
            for task in [t for t in tasks if t["pair_id"] == pair]:
                episode = run_episode(scenario_from_task(task), runner.generate, max_steps=15,
                    record_event=lambda event: append_record(out / "agent_retry_events.jsonl", {"id": task["id"], **event}))
                verify_episode(episode)
                filename = f"retry_episode_{task['id']}.json"
                save(out / filename, episode)
                audit = audit_course_episode(task, episode)
                save(out / f"retry_audit_{task['id']}.json", audit)
                index = next(i for i, r in enumerate(report["episodes"]) if r["id"] == task["id"])
                report["episodes"][index] = {"id": task["id"], "status": episode["status"], "steps": episode["steps"],
                    "audit": audit, "replay_verified": True, "episode_file": filename, "selected_attempt": 2}
                if globally_stopped:
                    raise RuntimeError("Global resource guard stops retries")
        report.update(status="PASS_ASSIGNED_DEVELOPMENT_RUN_COMPLETE", elapsed_seconds=time.monotonic() - started,
                      memory=runner.memory())
        save(out / "report.json", report)
    except Exception:
        report.update(status="FAILED_OR_BLOCKED", elapsed_seconds=time.monotonic() - started,
                      error=traceback.format_exc())
        save(out / "report.json", report)
        raise


if __name__ == "__main__":
    main()
