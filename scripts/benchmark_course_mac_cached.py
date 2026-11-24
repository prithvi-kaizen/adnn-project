"""Run a bounded cached-model engineering check; preserve exact raw requests and traces."""

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

from course_llm_audit import ROOT, load_json, save, sha
from diagnose_ollama_tools import SCHEMA
from run_course_llm_audit_local import LOCAL_ENDPOINT, call, fingerprint

from src.agent.loop import run_episode, verify_episode
from src.data.course import scenario_from_task
from src.models.course_native import parse_stated, stated_messages
from src.models.ollama_dev import ResourceMonitor, append_jsonl, resource_sample, system_memory, utc

CONFIG = ROOT / "configs/course_mac_cached_smoke_v1.json"
TASKS = ROOT / "data/course_fast_track/v2/development_tasks.jsonl"
STATED = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["reversible", "partially_reversible", "irreversible"]}},
    "required": ["label"], "additionalProperties": False,
}


def guard(config, baseline, deadline):
    latest = system_memory()
    if time.monotonic() >= deadline:
        raise RuntimeError("Bounded model runtime exhausted")
    if latest["memory_free_percentage"] < config["min_free_memory_percent"]:
        raise RuntimeError("Memory pressure guard triggered")
    if latest["swap_used_bytes"] - baseline["swap_used_bytes"] > config["max_added_swap_bytes"]:
        raise RuntimeError("Swap-growth guard triggered")
    return latest


def summarize_logprobs(reply):
    entries = reply.get("logprobs") or []
    tokens = [entry.get("token") for entry in entries]
    values = [entry.get("logprob") for entry in entries]
    content = reply["message"]["content"]
    return {
        "returned_tokens": len(entries),
        "all_finite": bool(entries) and all(isinstance(value, (int, float)) and math.isfinite(value)
                                           for value in values),
        "joined_matches_response": all(isinstance(token, str) for token in tokens) and "".join(tokens) == content,
        "generated_count": reply.get("eval_count"),
    }


def run_model(model, config, directory, tasks):
    baseline = system_memory()
    deadline = time.monotonic() + config["per_model_seconds"]
    model_id = model["tag"]
    result = {"tag": model_id, "digest": model["digest"], "baseline_memory": baseline,
              "requests": [], "stated": [], "episodes": [], "status": "ATTEMPTED"}
    owned = False

    def generate(messages, token_cap, purpose, schema=None, logprobs=False):
        nonlocal owned
        memory_before = guard(config, baseline, deadline)
        prompt_bytes = len(json.dumps(messages, ensure_ascii=False).encode())
        if prompt_bytes + 256 + token_cap > config["options"]["num_ctx"]:
            raise ValueError("Conservative context preflight failed")
        request = {"model": model_id, "messages": messages, "stream": False, "keep_alive": "2m",
                   "options": {**config["options"], "num_predict": token_cap}}
        if "think" in model:
            request["think"] = model["think"]
        if schema is not None:
            request["format"] = schema
        if logprobs:
            request.update(logprobs=True, top_logprobs=1)
        index = len(result["requests"]) + 1
        base = directory / f"{index:02}"
        save(base.with_suffix(".request.json"), request)
        started = time.monotonic()
        record = {"purpose": purpose, "request": index, "memory_before": memory_before, "status": "ATTEMPTED"}
        result["requests"].append(record)
        try:
            owned = True
            raw, reply = call("chat", request, timeout=config["request_seconds"])
            base.with_suffix(".response.json").write_bytes(raw)
            after = resource_sample()
            for key in ("prompt_eval_count", "eval_count", "eval_duration", "load_duration", "done_reason"):
                record[key] = reply.get(key)
            if reply.get("done") is not True:
                raise ValueError("Local runtime did not complete the response")
            record["sample_after"] = after
            record["memory_after"] = guard(config, baseline, deadline)
            record["status"] = "RETURNED"
            return reply
        except Exception as error:
            record.update(status="FAILED", error=str(error))
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            append_jsonl(directory / "requests.jsonl", record)

    try:
        with ResourceMonitor(directory / "resources.jsonl"):
            for task in tasks:
                response = generate(stated_messages(task), config["stated_token_cap"], "stated", STATED)
                label = parse_stated(response["message"]["content"]) if response["done_reason"] == "stop" else None
                result["stated"].append({"id": task["id"], "parsed_label": label, "oracle": task["label"],
                                         "correct": label == task["label"]})
            for task in tasks:
                def agent_call(messages):
                    response = generate(messages, config["agent_token_cap"], "agent", SCHEMA)
                    return {"text": response["message"]["content"], "done_reason": response["done_reason"],
                            "request": len(result["requests"])}
                episode = run_episode(scenario_from_task(task), agent_call, max_steps=config["agent_step_cap"],
                                      record_event=lambda event: append_jsonl(directory / f"{task['id']}.jsonl", event))
                verify_episode(episode)
                save(directory / f"{task['id']}.json", episode)
                result["episodes"].append({"id": task["id"], "terminal_status": episode["status"],
                                           "goal": episode["goal_met_at_termination"], "steps": episode["steps"],
                                           "executed": sum(row["status"] == "executed" for row in episode["trace"]),
                                           "rejected": sum(row["status"] == "rejected" for row in episode["trace"]),
                                           "replay_verified": True})
                if episode["status"] == "infrastructure_failure":
                    raise RuntimeError("Resource or service failure interrupted the episode")
            toy = config["logprob_probe"]
            try:
                reply = generate([{"role": "user", "content": toy["prompt"]}], toy["num_predict"],
                                 "logprob_sketch", logprobs=True)
                result["logprob_sketch"] = summarize_logprobs(reply)
            except Exception as error:
                result["logprob_sketch"] = {"error": str(error)}
        result["status"] = "COMPLETE_DEVELOPMENT_CHECK"
    except Exception as error:
        result.update(status="STOPPED", error=str(error))
    finally:
        try:
            if owned and any(row["name"] == model_id for row in call("ps")[1]["models"]):
                call("generate", {"model": model_id, "keep_alive": 0, "stream": False})
            result["resident_after"] = [row["name"] for row in call("ps")[1]["models"]]
        except Exception as error:
            result["unload_error"] = str(error)
        result["end_memory"] = system_memory()
        save(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/exploratory"):
        parser.error("Write benchmark records under artifacts/exploratory")
    config = load_json(args.config)
    if config["endpoint"] != LOCAL_ENDPOINT or call("version")[1]["version"] != config["ollama_version"]:
        parser.error("Unexpected local runtime")
    if call("ps")[1]["models"]:
        parser.error("Existing resident model; preserve its session")
    installed = {item["name"]: item for item in call("tags")[1]["models"]}
    for model in config["models"]:
        if installed.get(model["tag"], {}).get("digest") != model["digest"]:
            parser.error("The pinned cached model is unavailable")
    requests = config["fixtures"] if "fixtures" in config else [config["fixture"]]
    tasks = [row for line in TASKS.read_text().splitlines()
             if (row := json.loads(line))["pair_id"] in requests]
    if len(tasks) != 2 * len(requests) or {task["pair_id"] for task in tasks} != set(requests):
        raise ValueError("Each requested development pair must have two versions")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "scope": config["scope"], "started_utc": utc(), "platform": platform.platform(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "logical_cpu_count": os.cpu_count(),
        "unified_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
        "free_storage_bytes": shutil.disk_usage(ROOT).free, "starting_memory": system_memory(),
        "config_sha256": sha(args.config), "runner_sha256": sha(Path(__file__)), "task_source_sha256": sha(TASKS),
        "holdout_access": False, "weights_downloaded": False,
    }
    save(output / "run.json", metadata)
    for index, model in enumerate(config["models"], 1):
        directory = output / f"model_{index}"
        directory.mkdir()
        _, details = call("show", {"model": model["tag"]})
        save(directory / "model.json", {"installed": installed[model["tag"]],
                                      "template_sha256": fingerprint(details.get("template", ""))})
        result = run_model(model, config, directory, tasks)
        print(f"{model['tag']}: {result['status']}", flush=True)
        if result.get("unload_error") or result.get("resident_after"):
            break
        if result["end_memory"]["swap_used_bytes"] - metadata["starting_memory"]["swap_used_bytes"] > config["max_added_swap_bytes"]:
            break


if __name__ == "__main__":
    main()
