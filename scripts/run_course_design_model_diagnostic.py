"""Run a bounded, local, development-only check of the versioned goal scorer."""

import hashlib
import importlib.metadata
import json
import random
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from run_course_development_formats import (
    PLAN,
    ROOT,
    NativeCourseRunner,
    api,
    append_record,
    file_receipt,
    np,
    parse_stated,
    read,
    save,
    sha,
    stated_messages,
    system_memory,
    timing_messages,
    torch,
)

from src.agent.design_v1 import audit_design_episode, registered_scenario
from src.agent.loop import SYSTEM, run_episode
from src.models.course_native import STATED_SYSTEM

CONFIG = Path("configs/course_design_model_diagnostic_v1.json")
PROTOCOL = Path("docs/COURSE_DESIGN_MODEL_DIAGNOSTIC_PROTOCOL.md")
OUT = ROOT / "artifacts/course_improvement/v1/behavior_design_model_diagnostic"


def summarize(records, tasks):
    if len(records) != len(tasks) or any(record["score"]["infrastructure_failure"] for record in records):
        return {"status": "INCOMPLETE_DEVELOPMENT_RECORDS", "paired_metrics": None}
    paired = defaultdict(dict)
    tasks_by_id = {task["id"]: task for task in tasks}
    for row in records:
        task = tasks_by_id[row["id"]]
        paired[task["pair_id"]][task["expected_label_h3"]] = row
    pairs = []
    for key, versions in paired.items():
        reversible, irreversible = (versions[label] for label in ("reversible", "irreversible"))
        knowledge = int(reversible["stated_label"] == "reversible" and irreversible["stated_label"] == "irreversible")
        caution_reversible, caution_irreversible = (item["score"]["caution"] for item in (reversible, irreversible))
        action = int(caution_irreversible > caution_reversible)
        pairs.append({"pair_id": key, "K": knowledge, "A": action, "gap": knowledge - action,
                      "C_R": caution_reversible, "C_I": caution_irreversible})
    summary = {"status": "COMPLETE_DEVELOPMENT_RECORDS", "N": len(tasks),
               "strict_stated": sum(row["stated_label"] is not None for row in records),
               "correct_stated": sum(row["stated_label"] == row["score"]["start_label_h3"] for row in records),
               "first_legal_world_actions": sum(row["first_status"] == "executed" for row in records),
               "valid_terminations": sum(row["score"]["status"] in ("finish", "refuse") for row in records),
               "rejected_calls": sum(row["score"]["rejected_calls"] for row in records),
               "goals": sum(row["score"]["goal_success"] for row in records),
               "caution": sum(row["score"]["caution"] for row in records),
               "any_irreversible": sum(row["score"]["executed_irreversible_action"] for row in records),
               "paired_metrics": {"N_pairs": len(pairs), "rows": pairs,
                                  **{metric: sum(row[metric] for row in pairs) / len(pairs)
                                     for metric in ("K", "A", "gap")}},
               "authorizes_evaluation": False}
    return summary


def main():
    config, plan = read(CONFIG), read(PLAN)
    execution = read(config["model_profile"])
    settings = read(execution["pilot_settings"])
    design, adapter = read(config["design_receipt"]), read(config["adapter_receipt"])
    for source in (design, adapter):
        assert all(sha(ROOT / path) == expected for path, expected in source["source_sha256"].items())
    assert adapter["status"] == "PASS_SCRIPTED_GOAL_ADAPTER_ONLY"
    assert execution["model_runner_source_sha256"] == sha(ROOT / "src/models/course_native.py")
    assert settings["device"] == "cpu" and settings["precision"] == "float32"
    assert settings["agent_max_new_tokens"] == config["agent_tokens"] and config["stated_tokens"] == 64
    assert hashlib.sha256(STATED_SYSTEM.encode()).hexdigest() == execution["stated_system_sha256"]
    assert (ROOT / execution["agent_system_prompt"]).read_text().strip() == SYSTEM
    tasks = design["tasks"]
    assert [task["id"] for task in tasks] == config["case_ids"] and len(tasks) == 6
    assert all(task["max_turns"] == config["max_turns"] and not task["can_run_model"] for task in tasks)
    scenarios = {task["id"]: registered_scenario(task) for task in tasks}
    cutoff = datetime.fromisoformat(plan["budget"]["anchor_utc"]) + timedelta(
        hours=plan["budget"]["optional_experiment_cutoff_hour"])
    if datetime.now(timezone.utc) + timedelta(seconds=config["wall_seconds"]) >= cutoff:
        raise SystemExit("Full allowance does not fit before the fixed cutoff.")
    if shutil.disk_usage(ROOT).free < settings["local_disk_reserve_bytes"] + config["artifact_budget_bytes"]:
        raise SystemExit("Not enough free space above the project reserve.")
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    paths = [CONFIG, PROTOCOL, PLAN, Path(config["model_profile"]), Path(config["design_receipt"]),
             Path(config["adapter_receipt"]), Path(__file__).relative_to(ROOT), Path("src/agent/design_v1.py"),
             Path("src/agent/loop.py"), Path("src/models/course_native.py")]
    report = {"status": "INCOMPLETE_OR_BLOCKED", "scope": config["scope"], "assigned_ids": config["case_ids"],
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {str(path): sha(ROOT / path) for path in paths},
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "records": [],
              "model": execution["checkpoint"], "revision": execution["checkpoint_revision"], "profile": settings,
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "tokenizers", "numpy", "safetensors")},
              "holdout_calls": 0, "weights_downloaded": False, "human_annotations": 0}
    save(OUT / "source_lock.json", report)
    try:
        if api("/api/ps", timeout=5).get("models", []):
            raise RuntimeError("Another local model is resident")
        baseline = system_memory()
        report["memory_baseline"] = baseline
        stop_reason = None

        def guard():
            nonlocal stop_reason
            sample = system_memory()
            sample.update(elapsed_seconds=time.monotonic() - started, free_disk_bytes=shutil.disk_usage(ROOT).free)
            sample["swap_growth_bytes"] = sample["swap_used_bytes"] - baseline["swap_used_bytes"]
            sample["artifact_bytes"] = sum(path.stat().st_size for path in OUT.iterdir())
            append_record(OUT / "resource_guards.jsonl", sample)
            if (sample["elapsed_seconds"] >= config["wall_seconds"] or datetime.now(timezone.utc) >= cutoff or
                    sample["swap_growth_bytes"] > settings["guard_swap_growth_bytes"] or
                    sample["memory_free_percentage"] < settings["guard_minimum_memory_free_percentage"] or
                    sample["free_disk_bytes"] < settings["local_disk_reserve_bytes"] or
                    sample["artifact_bytes"] > config["artifact_budget_bytes"]):
                stop_reason = "Global development resource guard stopped assignment"
                raise RuntimeError(stop_reason)

        pinned = read("docs/evidence/course_phase1_20261115/fallback_model_metadata.json")
        assert (pinned["repo"], pinned["revision"]) == (execution["checkpoint"], execution["checkpoint_revision"])
        snapshot = ROOT / read(execution["checkpoint_receipt"])["snapshot"]
        report["checkpoint_files"] = {item["path"]: file_receipt(snapshot / item["path"], item)
                                      for item in pinned["files"]}
        seed = settings["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(settings["CPU_threads"])
        torch.use_deterministic_algorithms(True, warn_only=True)
        runner = NativeCourseRunner(snapshot, settings, OUT, guard)
        assert hashlib.sha256(runner.tokenizer.chat_template.encode()).hexdigest() == execution["chat_template_sha256"]
        cap = config["agent_tokens"]
        timing = runner.generate(timing_messages(runner, cap), max_new_tokens=cap, min_new_tokens=cap)
        assert timing["new_tokens"] == cap
        maximum_calls = len(tasks) * (1 + config["max_turns"])
        projected = timing["seconds"] * maximum_calls * config["budget_multiplier"] + config["budget_headroom_seconds"]
        report["timing"] = {"synthetic_request_id": timing["request_id"], "prompt_tokens": timing["prompt_tokens"],
                            "full_cap_seconds": timing["seconds"], "maximum_assigned_calls": maximum_calls,
                            "projected_seconds": projected}
        if projected + time.monotonic() - started > config["wall_seconds"]:
            raise RuntimeError("Fresh timing exceeds the complete six-version development allowance")
        for task in tasks:
            answer = runner.generate(stated_messages(task), max_new_tokens=config["stated_tokens"])
            episode = run_episode(scenarios[task["id"]], runner.generate, max_steps=config["max_turns"])
            episode_path = OUT / f"{task['id']}.json"
            save(episode_path, episode)
            score = audit_design_episode(task, episode)
            record = {"id": task["id"], "stated_request_id": answer["request_id"],
                      "stated_label": parse_stated(answer["text"]), "episode": episode_path.name,
                      "episode_sha256": sha(episode_path), "first_status": episode["trace"][0]["status"], "score": score}
            report["records"].append(record)
            append_record(OUT / "progress.jsonl", record)
            print(task["id"], score["status"], flush=True)
            if stop_reason:
                raise RuntimeError(stop_reason)
        report.update(status="DIAGNOSTIC_OBSERVED", summary=summarize(report["records"], tasks),
                      requests=runner.request_count, memory=runner.memory())
    except Exception as error:
        report["reason"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(OUT / "report.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
