"""Run or verify a single cached model on the six exposed safer-route fixtures."""

import argparse
import http.client
import json
import platform
import shutil
import socket
import threading
import time
from pathlib import Path

from benchmark_course_mac_cached import STATED
from course_llm_audit import ROOT, load_json, save, sha
from diagnose_ollama_tools import SCHEMA
from run_course_llm_audit_local import call, fingerprint

from src.agent.design_v1 import audit_design_episode, registered_scenario
from src.agent.loop import run_episode
from src.models.course_native import parse_stated, stated_messages
from src.models.ollama_dev import append_jsonl, resource_sample, system_memory, utc

CONFIG = ROOT / "configs/mac_safer_behavior_v1.json"


class MacBudget:
    """Sample system memory during decoding and interrupt our own in-flight socket."""

    def __init__(self, config, directory):
        self.config = config
        self.directory = directory
        self.baseline = system_memory()
        self.started = time.monotonic()
        self.failure = None
        self.active = None
        self.stop = threading.Event()
        self.monitor = threading.Thread(target=self.watch, daemon=True)

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)
        current = system_memory()
        current.update(at_utc=utc(), free_disk_bytes=shutil.disk_usage(self.directory).free,
                       added_swap_bytes=current["swap_used_bytes"] - self.baseline["swap_used_bytes"],
                       elapsed_seconds=time.monotonic() - self.started)
        append_jsonl(self.directory / "memory.jsonl", current)
        if current["memory_free_percentage"] < self.config["min_free_memory_percent"]:
            raise RuntimeError("Mac memory-pressure limit reached")
        if current["added_swap_bytes"] > self.config["max_added_swap_bytes"]:
            raise RuntimeError("Mac swap-growth limit reached")
        if current["free_disk_bytes"] < self.config["min_free_disk_bytes"]:
            raise RuntimeError("Mac disk reserve reached")
        if current["elapsed_seconds"] > self.config["wall_seconds"]:
            raise RuntimeError("Model wall-time limit reached")
        return current

    def watch(self):
        while not self.stop.wait(self.config["sample_seconds"]):
            try:
                self.check()
            except Exception as error:
                self.failure = str(error)
                active_socket = self.active
                if active_socket is not None:
                    try:
                        active_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                return

    def __enter__(self):
        self.check()
        self.monitor.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.monitor.join(timeout=5)

    def chat(self, request):
        self.check()
        connection = http.client.HTTPConnection("127.0.0.1", 11434, timeout=self.config["request_seconds"])
        try:
            connection.connect()
            self.active = connection.sock
            if self.failure:
                raise RuntimeError(self.failure)
            connection.request("POST", "/api/chat", body=json.dumps(request).encode(),
                               headers={"Content-Type": "application/json"})
            reply = connection.getresponse()
            return reply.status, reply.read(1 << 20)
        finally:
            connection.close()
            self.active = None


def load_design(config):
    path = ROOT / config["design_receipt"]
    receipt = load_json(path)
    assert receipt["status"] == "PASS_SCRIPTED_BEHAVIOR_FIXTURES_ONLY"
    assert all(sha(ROOT / name) == expected for name, expected in receipt["source_sha256"].items())
    tasks = receipt["tasks"]
    assert len({t["pair_id"] for t in tasks}) == 3 and len(tasks) == 6
    for task in tasks:
        scenario = registered_scenario(task)
        safe = receipt["receipts"][scenario.scenario_id]["safer"]
        assert safe["goal_met_at_finish"] and safe["turns_including_finish"] <= task["max_turns"]
        assert all(e["labels_by_horizon"]["3"]["label"] == "reversible" for e in safe["events"])
    return path, tasks


def run(config, model, design, tasks, directory):
    assert call("version")[1]["version"] == config["ollama_version"]
    if call("ps")[1]["models"]:
        raise RuntimeError("An existing Ollama model is resident. Preserve its session.")
    tag = model["tag"]
    installed = {row["name"]: row for row in call("tags")[1]["models"]}
    if installed.get(tag, {}).get("digest") != model["digest"]:
        raise RuntimeError("The pinned local checkpoint is missing")
    directory.mkdir(parents=True, exist_ok=False)
    shown = call("show", {"model": tag})[1]
    save(directory / "run.json", {"configuration": config, "model": installed[tag],
         "template_sha256": fingerprint(shown["template"]), "runner_sha256": sha(Path(__file__)),
         "design_sha256": sha(design), "assigned_ids": [t["id"] for t in tasks], "started_utc": utc(),
         "platform": platform.platform(), "holdout_access": False, "downloaded_weights": False})
    budget = MacBudget(config, directory)
    requests = []

    def generate(task, purpose, messages, schema, cap):
        if len(json.dumps(messages, ensure_ascii=False).encode()) + 256 + cap > config["options"]["num_ctx"]:
            raise RuntimeError("Context preflight limit")
        i = len(requests) + 1
        request = {"model": tag, "messages": messages, "stream": False, "format": schema, "keep_alive": "90s",
                   "options": {**config["options"], "num_predict": cap}}
        if "think" in model:
            request["think"] = model["think"]
        save(directory / f"{i:03d}.request.json", request)
        row = {"request": i, "case_id": task["id"], "purpose": purpose, "status": "ATTEMPTED"}
        requests.append(row)
        before = time.monotonic()
        try:
            status, raw = budget.chat(request)
            (directory / f"{i:03d}.response.json").write_bytes(raw)
            budget.check()
            if status != 200:
                raise RuntimeError(f"Ollama HTTP {status}")
            reply = json.loads(raw)
            if reply.get("done") is not True or reply.get("model") != tag:
                raise RuntimeError("Incomplete or wrong-model reply")
            row.update(status="RETURNED", done_reason=reply.get("done_reason"), rss=resource_sample())
            return {"text": reply["message"]["content"], "done_reason": reply.get("done_reason"), "request": i}
        except Exception as error:
            row.update(status="FAILED", error=budget.failure or str(error))
            raise RuntimeError(row["error"]) from error
        finally:
            row["elapsed_seconds"] = time.monotonic() - before
            append_jsonl(directory / "requests.jsonl", row)

    result = {"status": "STOPPED", "baseline": budget.baseline, "stated": {}, "episode_ids": []}
    try:
        with budget:
            for task in tasks:
                key = task["id"]
                label = generate(task, "stated", stated_messages(task), STATED, config["stated_token_cap"])
                result["stated"][key] = label["request"]
            for task in tasks:
                key = task["id"]
                scenario = registered_scenario(task)

                def agent(messages):
                    return generate(task, "agent", messages, SCHEMA, config["agent_token_cap"])

                episode = run_episode(scenario, agent, max_steps=task["max_turns"],
                                      record_event=lambda event: append_jsonl(directory / f"{key}.jsonl", event))
                save(directory / f"{key}.json", episode)
                result["episode_ids"].append(key)
                print(f"{tag} {key} {episode['status']}", flush=True)
                if episode["status"] == "infrastructure_failure":
                    raise RuntimeError(episode["trace"][-1]["error"])
            result["status"] = "COMPLETE_DEVELOPMENT_RUN"
    except Exception as error:
        result["reason"] = str(error)
    finally:
        try:
            if requests and any(m["name"] == tag for m in call("ps")[1]["models"]):
                call("generate", {"model": tag, "keep_alive": 0, "stream": False})
            result["resident_after"] = [m["name"] for m in call("ps")[1]["models"]]
        except Exception as error:
            result["unload_error"] = str(error)
        result["request_count"] = len(requests)
        result["elapsed_seconds"] = time.monotonic() - budget.started
        save(directory / "result.json", result)
    if result["status"] != "COMPLETE_DEVELOPMENT_RUN" or result.get("resident_after") != []:
        raise SystemExit(json.dumps(result))


def verify(directory, config, design, tasks):
    meta, result = (load_json(directory / f"{name}.json") for name in ("run", "result"))
    assert meta["configuration"] == config and sha(design) == meta["design_sha256"]
    assert not meta["holdout_access"] and not meta["downloaded_weights"]
    assert meta["assigned_ids"] == [t["id"] for t in tasks]
    model = next(item for item in config["models"].values() if item["tag"] == meta["model"]["name"])
    assert model["digest"] == meta["model"]["digest"]
    ledger = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    assert len(ledger) == result["request_count"]

    def response(i, task, purpose, messages, schema, cap):
        row = ledger[i - 1]
        assert (row["request"], row["case_id"], row["purpose"], row["status"]) == (i, task, purpose, "RETURNED")
        req, reply = (load_json(directory / f"{i:03d}.{suffix}.json") for suffix in ("request", "response"))
        assert req["model"] == reply["model"] == model["tag"]
        assert req["messages"] == messages and req["format"] == schema
        assert req["options"] == dict(config["options"], num_predict=cap)
        assert req.get("think") == model.get("think")
        assert reply["done"] is True and reply["done_reason"] == row["done_reason"]
        return reply

    outputs = []
    for task in tasks:
        key = task["id"]
        label, score = None, None
        if key in result["stated"]:
            reply = response(result["stated"][key], key, "stated", stated_messages(task), STATED,
                             config["stated_token_cap"])
            if reply["done_reason"] == "stop":
                label = parse_stated(reply["message"]["content"])
        if key in result["episode_ids"]:
            episode = load_json(directory / f"{key}.json")
            assert episode["trace"] == list(map(json.loads, (directory / f"{key}.jsonl").read_text().splitlines()))
            score = audit_design_episode(task, episode)
            for event in episode["trace"]:
                if event["status"] == "infrastructure_failure":
                    continue
                reply = response(event["generation"]["request"], key, "agent", event["request_messages"], SCHEMA,
                                 config["agent_token_cap"])
                assert reply["message"]["content"] == event["raw_output"]
        outputs.append({"id": key, "pair_id": task["pair_id"], "oracle": task["expected_label_h3"],
                        "stated": label, "score": score})
    memory = list(map(json.loads, (directory / "memory.jsonl").read_text().splitlines()))
    return {"scope": config["scope"], "tag": model["tag"], "digest": model["digest"], "run": result,
            "minimum_free_memory_percent": min(m["memory_free_percentage"] for m in memory),
            "maximum_added_swap_bytes": max(m["added_swap_bytes"] for m in memory),
            "minimum_free_disk_bytes": min(m["free_disk_bytes"] for m in memory),
            "maximum_sampled_rss_bytes": max((r.get("rss", {}).get("sum_process_rss_bytes", 0) for r in ledger),
                                             default=0), "outputs": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", choices=("qwen", "llama"))
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    config = load_json(CONFIG)
    design, tasks = load_design(config)
    directory = args.directory.resolve()
    if not directory.is_relative_to(ROOT / "artifacts/exploratory"):
        parser.error("Use the exploratory artifact tree")
    if args.model:
        run(config, config["models"][args.model], design, tasks, directory)
    else:
        save(args.verify, verify(directory, config, design, tasks))
