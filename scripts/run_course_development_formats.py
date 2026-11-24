"""Run one bounded repeat of the baseline and one format-only candidate on old cases."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
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
from src.agent.loop import SYSTEM, run_episode, verify_episode  # noqa: E402
from src.data.course import scenario_from_task  # noqa: E402
from src.models.course_native import (  # noqa: E402
    STATED_SYSTEM,
    NativeCourseRunner,
    append_record,
    parse_stated,
    stated_messages,
)
from src.models.feature_store import sha  # noqa: E402
from src.models.ollama_dev import api, system_memory  # noqa: E402

CONFIG = Path("configs/course_development_format_v1.json")
PLAN = Path("configs/course_improvement_v1.json")
EXECUTION = Path("configs/course_execution_native_mac_cpu_v1.json")
DATA = Path("data/course_fast_track/v2")


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read(path):
    return json.loads((ROOT / path).read_text())


def system_prompt(config, kind, condition):
    if kind not in ("agent", "stated") or condition not in ("baseline", "candidate"):
        raise ValueError("An unregistered condition cannot be run")
    original = SYSTEM if kind == "agent" else STATED_SYSTEM
    if condition == "baseline":
        return original
    if kind == "agent":
        return config["agent_prefix"] + original
    anchor = config["stated_example_to_remove"]
    if original.count(anchor) != 1:
        raise ValueError("Stated anchor is no longer uniquely defined")
    return original.replace(anchor, "")


def with_system(messages, original, changed):
    if messages[0] != {"role": "system", "content": original}:
        raise ValueError("Unexpected system message; cannot apply locked format")
    return [{"role": "system", "content": changed}, *messages[1:]]


def summarize(config, episodes, stated, assigned):
    decisions = {}
    for condition in config["conditions"]:
        agent_rows = episodes[condition]
        stated_rows = stated[condition]
        if ([row["id"] for row in agent_rows] != assigned or [row["id"] for row in stated_rows] != assigned or
                any(row["status"] == "infrastructure_failure" for row in agent_rows)):
            return {"status": "INCOMPLETE_COMPARISON_NO_FORMAT_SELECTED"}
        first = sum(row["first_status"] == "executed" for row in agent_rows)
        terminal = sum(row["status"] in ("finish", "refuse") for row in agent_rows)
        goals = sum(row["goal"] for row in agent_rows)
        count = len(assigned)
        decisions[condition] = {
            "N": count, "legal_first_world_actions": first, "first_turn_terminal": sum(
                row["first_status"] == "terminal" for row in agent_rows),
            "valid_terminations": terminal, "goals_at_stop": goals,
            "strict_stated_parses": sum(row["label"] is not None for row in stated_rows),
            "stated_correct_diagnostic_only": sum(row["label"] == row["truth"] for row in stated_rows),
            "agent_qualified": first / count >= config["first_legal_world_action_min"] and
            terminal / count >= config["valid_terminal_min"] and goals / count >= config["goal_at_stop_min"]}
    agent = next((condition for condition in ("candidate", "baseline") if decisions[condition]["agent_qualified"]),
                 None)
    stated_choice = ("candidate" if decisions["candidate"]["strict_stated_parses"] >=
                     decisions["baseline"]["strict_stated_parses"] else "baseline")
    return {"status": "DEVELOPMENT_COMPARISON_COMPLETE", "conditions": decisions,
            "agent_choice": agent, "stated_choice": stated_choice, "authorizes_holdout": False}


def timing_messages(runner, token_cap):
    words = 1000
    for _ in range(16):
        if words <= 0:
            break
        messages = [{"role": "system", "content": "Synthetic timing request only."},
                    {"role": "user", "content": "Count audit markers." + " audit" * words}]
        try:
            _, encoded = runner.encode(messages)
        except ValueError:
            words -= 128
            continue
        length = len(encoded["input_ids"])
        margin = runner.settings["context_limit_including_generation"] - token_cap - length
        if margin >= 0 and margin <= 16:
            return messages
        words += margin
    raise RuntimeError("Cannot build the declared near-context timing request")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output
    config, plan, execution = read(CONFIG), read(PLAN), read(EXECUTION)
    settings = read(execution["pilot_settings"])
    cutoff = datetime.fromisoformat(plan["budget"]["anchor_utc"]) + timedelta(
        hours=plan["budget"]["optional_experiment_cutoff_hour"])
    if datetime.now(timezone.utc) + timedelta(seconds=config["pilot_wall_seconds"]) >= cutoff:
        raise SystemExit("Full diagnostic allowance does not fit before experiment cutoff")
    assert sys.platform == "darwin" and settings["device"] == "cpu"
    assert sha(ROOT / "src/models/course_native.py") == execution["model_runner_source_sha256"]
    frozen = read(DATA / "freeze_manifest.json")
    assert sha(ROOT / DATA / "freeze_manifest.json") == plan["protected_baseline_sha256"][
        str(DATA / "freeze_manifest.json")]
    assert all(sha(ROOT / DATA / name) == entry["sha256"] for name, entry in frozen["files"].items())
    tasks = [json.loads(line) for line in (ROOT / DATA / "development_tasks.jsonl").read_text().splitlines()]
    tasks = [row for row in tasks if row["pair_id"] in config["pair_ids"]]
    assert len(tasks) == 6 and len({row["pair_id"] for row in tasks}) == 3
    assert all(row["RI_gap_eligible"] for row in tasks)
    assigned = [row["id"] for row in tasks]
    prompts = {kind: {condition: system_prompt(config, kind, condition) for condition in config["conditions"]}
               for kind in ("agent", "stated")}
    assert hashlib.sha256(prompts["stated"]["baseline"].encode()).hexdigest() == execution["stated_system_sha256"]
    assert prompts["agent"]["baseline"] == (ROOT / execution["agent_system_prompt"]).read_text().strip()
    paths = [CONFIG, PLAN, EXECUTION, Path(execution["pilot_settings"]), DATA / "development_tasks.jsonl",
             Path(__file__).resolve().relative_to(ROOT), Path("docs/COURSE_DEVELOPMENT_FORMAT_AMENDMENT.md"),
             Path("src/models/course_native.py"), Path("src/agent/loop.py"),
             Path("scripts/run_course_native_pilot.py")]
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"scope": config["scope"], "model": execution["checkpoint"], "revision": execution["checkpoint_revision"],
              "profile": settings, "assigned_ids": assigned, "start_utc": datetime.now(timezone.utc).isoformat(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {str(path): sha(ROOT / path) for path in paths}, "system_prompts": prompts,
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "tokenizers", "numpy", "safetensors")},
              "holdout_calls": 0, "weights_downloaded": False, "human_annotations": 0,
              "episodes": {key: [] for key in config["conditions"]},
              "stated": {key: [] for key in config["conditions"]}}
    save(out / "source_lock.json", report)
    try:
        if api("/api/ps", timeout=5).get("models", []):
            raise RuntimeError("Another Ollama checkpoint is resident")
        baseline = system_memory()
        report["memory_baseline"] = baseline
        stop_reason = None

        def guard():
            nonlocal stop_reason
            measured = system_memory()
            measured.update(elapsed_seconds=time.monotonic() - started, free_disk_bytes=shutil.disk_usage(ROOT).free)
            measured["swap_growth_bytes"] = measured["swap_used_bytes"] - baseline["swap_used_bytes"]
            measured["artifact_bytes"] = sum(path.stat().st_size for path in out.iterdir())
            append_record(out / "resource_guards.jsonl", measured)
            if (measured["elapsed_seconds"] > config["pilot_wall_seconds"] or
                    datetime.now(timezone.utc) >= cutoff or
                    measured["swap_growth_bytes"] > settings["guard_swap_growth_bytes"] or
                    measured["memory_free_percentage"] < settings["guard_minimum_memory_free_percentage"] or
                    measured["free_disk_bytes"] < settings["local_disk_reserve_bytes"] or
                    measured["artifact_bytes"] > config["output_budget_bytes"]):
                stop_reason = "Global development runtime, disk or memory guard failed"
                raise RuntimeError(stop_reason)

        if shutil.disk_usage(ROOT).free < settings["local_disk_reserve_bytes"] + config["output_budget_bytes"]:
            raise RuntimeError("Development artifacts do not fit above disk reserve")
        for task in tasks:
            pilot_exposure(out, task, ROOT / DATA, execution)
        receipt = read(execution["checkpoint_receipt"])
        snapshot = ROOT / receipt["snapshot"]
        metadata = read("docs/evidence/course_phase1_20261115/fallback_model_metadata.json")
        assert (metadata["repo"], metadata["revision"]) == (execution["checkpoint"], execution["checkpoint_revision"])
        report["checkpoint_files"] = {item["path"]: file_receipt(snapshot / item["path"], item)
                                      for item in metadata["files"]}
        seed = settings["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(settings["CPU_threads"])
        torch.use_deterministic_algorithms(True, warn_only=True)
        runner = NativeCourseRunner(snapshot, settings, out, guard)
        assert hashlib.sha256(runner.tokenizer.chat_template.encode()).hexdigest() == execution["chat_template_sha256"]
        cap = settings["agent_max_new_tokens"]
        timing = runner.generate(timing_messages(runner, cap), max_new_tokens=cap, min_new_tokens=cap)
        assert timing["new_tokens"] == cap
        calls = len(tasks) * len(config["conditions"]) * (15 + 1)
        projected = config["runtime_multiplier"] * calls * timing["seconds"] + config["headroom_seconds"]
        report["timing"] = {"synthetic_only": True, "request_id": timing["request_id"],
                            "full_cap_seconds": timing["seconds"], "prompt_tokens": timing["prompt_tokens"],
                            "full_cap_tokens": cap, "maximum_assigned_calls": calls,
                            "projected_remaining_seconds": projected}
        if projected + time.monotonic() - started > config["pilot_wall_seconds"]:
            raise RuntimeError("Worst-case estimate does not fit the complete assigned comparison")
        for condition in config["conditions"]:
            for task in tasks:
                stated_response = runner.generate(with_system(
                    stated_messages(task), STATED_SYSTEM, prompts["stated"][condition]),
                    max_new_tokens=settings["stated_max_new_tokens"])
                report["stated"][condition].append({"id": task["id"], "label": parse_stated(stated_response["text"]),
                                                   "truth": task["label"], "request_id": stated_response["request_id"]})
                episode = run_episode(scenario_from_task(task), lambda messages: runner.generate(
                    with_system(messages, SYSTEM, prompts["agent"][condition])), max_steps=15)
                verify_episode(episode)
                filename = f"{condition}_{task['id']}.json"
                save(out / filename, episode)
                report["episodes"][condition].append({
                    "id": task["id"], "status": episode["status"], "goal": episode["goal_met_at_termination"],
                    "first_status": episode["trace"][0]["status"], "steps": episode["steps"],
                    "rejected": sum(item["status"] == "rejected" for item in episode["trace"]),
                    "file": filename, "sha256": sha(out / filename)})
                print(f"{condition} {task['id']}: {episode['status']}", flush=True)
                if stop_reason is not None:
                    raise RuntimeError(stop_reason)
        report.update(status="RAN_ASSIGNED_DEVELOPMENT_COMPARISON",
                      decision=summarize(config, report["episodes"], report["stated"], assigned),
                      request_count=runner.request_count, memory=runner.memory())
    except Exception as error:
        report.update(status="PARTIAL_OR_BLOCKED", reason=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(out / "report.json", report)


if __name__ == "__main__":
    main()
