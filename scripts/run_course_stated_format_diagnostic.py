"""Compare only the locked stated formats after the agent workload misses its budget."""

import hashlib
import json
import random
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from run_course_development_formats import (
    CONFIG,
    DATA,
    EXECUTION,
    PLAN,
    ROOT,
    STATED_SYSTEM,
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
    torch,
    with_system,
)

BLOCKED = Path("artifacts/course_improvement/v1/development_formats")
OUT = ROOT / "artifacts/course_improvement/v1/stated_format_only"
AMENDMENT = Path("docs/COURSE_STATED_FORMAT_DIAGNOSTIC.md")
MAX_SECONDS = 300


def main():
    combined = read(BLOCKED / "report.json")
    assert combined["reason"] == "Worst-case estimate does not fit the complete assigned comparison"
    assert not any(combined[kind][condition] for kind in ("episodes", "stated")
                   for condition in ("baseline", "candidate"))
    assert all(sha(ROOT / path) == checksum for path, checksum in combined["source_sha256"].items())
    config, execution, plan = read(CONFIG), read(EXECUTION), read(PLAN)
    cutoff = datetime.fromisoformat(plan["budget"]["anchor_utc"]) + timedelta(
        hours=plan["budget"]["optional_experiment_cutoff_hour"])
    if datetime.now(timezone.utc) + timedelta(seconds=MAX_SECONDS) >= cutoff:
        raise SystemExit("The five-minute stated diagnostic does not fit before the cutoff.")
    settings = combined["profile"]
    assert settings == read(execution["pilot_settings"]) and settings["device"] == "cpu"
    tasks = [json.loads(line) for line in (ROOT / DATA / "development_tasks.jsonl").read_text().splitlines()]
    tasks = [row for row in tasks if row["id"] in combined["assigned_ids"]]
    assert [row["id"] for row in tasks] == combined["assigned_ids"]
    first, second = config["conditions"]
    prompts = combined["system_prompts"]["stated"]
    assert prompts[first] == STATED_SYSTEM
    assert prompts[second] == STATED_SYSTEM.replace(config["stated_example_to_remove"], "")
    projection = config["runtime_multiplier"] * len(tasks) * 2 * combined["timing"]["full_cap_seconds"]
    projection += config["headroom_seconds"]
    assert projection < MAX_SECONDS
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "PARTIAL_OR_BLOCKED", "scope": "previously_exposed_stated_format_diagnostic_only",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {str(path): sha(ROOT / path) for path in (
                  BLOCKED / "report.json", BLOCKED / "pilot_exposures.jsonl", AMENDMENT,
                  Path(__file__).relative_to(ROOT), Path("src/models/course_native.py"), CONFIG, EXECUTION)},
              "model": combined["model"], "revision": combined["revision"], "profile": settings,
              "packages": combined["packages"], "assigned_ids": combined["assigned_ids"],
              "system_prompts": prompts, "projection_seconds": projection,
              "holdout_calls": 0, "weights_downloaded": False, "agent_episodes": 0, "records": []}
    save(OUT / "source_lock.json", report)
    try:
        if api("/api/ps", timeout=5).get("models", []):
            raise RuntimeError("Another local model is resident")
        baseline = system_memory()
        report["memory_baseline"] = baseline

        def guard():
            memory = system_memory()
            memory.update(elapsed_seconds=time.monotonic() - started, free_disk_bytes=shutil.disk_usage(ROOT).free)
            memory["swap_growth_bytes"] = memory["swap_used_bytes"] - baseline["swap_used_bytes"]
            if (memory["elapsed_seconds"] > MAX_SECONDS or datetime.now(timezone.utc) >= cutoff or
                    memory["swap_growth_bytes"] > settings["guard_swap_growth_bytes"] or
                    memory["memory_free_percentage"] < settings["guard_minimum_memory_free_percentage"] or
                    memory["free_disk_bytes"] < settings["local_disk_reserve_bytes"]):
                raise RuntimeError("Stated diagnostic hit a resource guard")
            append_record(OUT / "resource_guards.jsonl", memory)

        seed = settings["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(settings["CPU_threads"])
        torch.use_deterministic_algorithms(True, warn_only=True)
        receipt = read(execution["checkpoint_receipt"])
        snapshot = ROOT / receipt["snapshot"]
        pinned = read("docs/evidence/course_phase1_20261115/fallback_model_metadata.json")
        report["checkpoint_files"] = {item["path"]: file_receipt(snapshot / item["path"], item)
                                      for item in pinned["files"]}
        assert report["checkpoint_files"] == combined["checkpoint_files"]
        runner = NativeCourseRunner(snapshot, settings, OUT, guard)
        assert hashlib.sha256(runner.tokenizer.chat_template.encode()).hexdigest() == execution["chat_template_sha256"]
        for condition in config["conditions"]:
            for task in tasks:
                messages = with_system(stated_messages(task), STATED_SYSTEM, prompts[condition])
                response = runner.generate(messages, max_new_tokens=settings["stated_max_new_tokens"])
                report["records"].append({"condition": condition, "id": task["id"], "truth": task["label"],
                                          "request_id": response["request_id"], "label": parse_stated(response["text"])})
        records = report["records"]
        counts = {condition: sum(row["label"] is not None for row in records if row["condition"] == condition)
                  for condition in config["conditions"]}
        report.update(status="COMPLETE_STATED_FORMAT_DIAGNOSTIC", strict_parse_counts=counts,
                      stated_choice=second if counts[second] >= counts[first] else first,
                      selected_for_holdout=False, memory=runner.memory())
        print(report["status"], counts, "development choice:", report["stated_choice"], flush=True)
    except Exception as error:
        report["reason"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(OUT / "report.json", report)


if __name__ == "__main__":
    main()
