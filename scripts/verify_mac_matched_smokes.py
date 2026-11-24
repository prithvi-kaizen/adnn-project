"""Reconstruct matched stated and behavior measures from cached-model development records."""

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmark_course_mac_cached import TASKS
from course_llm_audit import ROOT, load_json, save, sha

from src.agent.course_metrics import audit_course_episode
from src.agent.course_summary import behavior_summary
from src.data.behavior_readiness import certify_safe_route
from src.data.course import scenario_from_task
from src.models.course_native import parse_stated, stated_messages
from src.sandbox.labeling import label_action

RUNS = {"Llama": ((1, 1), (3, 1)), "Qwen": ((1, 2), (2, 1))}


def load_run(version, index, tasks):
    root = ROOT / f"artifacts/exploratory/mac_cached_smoke_v{version}"
    directory = root / f"model_{index}"
    manifest = load_json(root / "run.json")
    config_path = ROOT / f"configs/course_mac_cached_smoke_v{version}.json"
    config = load_json(config_path)
    identity = config["models"][index - 1]
    result = load_json(directory / "result.json")
    assert manifest["config_sha256"] == sha(config_path) and manifest["task_source_sha256"] == sha(TASKS)
    assert manifest["runner_sha256"] == sha(root / "runner_used.py")
    assert not manifest["holdout_access"] and not manifest["weights_downloaded"]
    assert not result["resident_after"] and result["digest"] == identity["digest"]
    assert load_json(directory / "model.json")["installed"]["digest"] == identity["digest"]
    target_pairs = config.get("fixtures", [config.get("fixture")])
    assigned = [t for t in tasks if t["pair_id"] in target_pairs]
    stated, audits, terminal, traces = {}, {}, {}, {}
    source_hashes = {str((directory / "result.json").relative_to(ROOT)): sha(directory / "result.json")}
    replies = {}
    pending = iter(assigned)
    for request_info in result["requests"]:
        number = request_info["request"]
        request_path = directory / f"{number:02}.request.json"
        response_path = directory / f"{number:02}.response.json"
        assert load_json(request_path)["model"] == result["tag"]
        if request_info["status"] != "RETURNED":
            continue
        request, reply = load_json(request_path), load_json(response_path)
        assert request["model"] == reply["model"] and reply["done"] is True
        assert all(request["options"][k] == v for k, v in config["options"].items())
        for path in (request_path, response_path):
            source_hashes[str(path.relative_to(ROOT))] = sha(path)
        replies[number] = (request, reply, request_info["purpose"])
        if request_info["purpose"] == "stated":
            task = next(pending)
            assert request["messages"] == stated_messages(task)
            stated[task["id"]] = parse_stated(reply["message"]["content"]) if reply["done_reason"] == "stop" else None
    for task in assigned:
        key = task["id"]
        trace_path = directory / f"{key}.json"
        if not trace_path.exists():
            continue
        episode = load_json(trace_path)
        path = directory / f"{key}.jsonl"
        assert episode["trace"] == list(map(json.loads, path.read_text().splitlines()))
        audit = audit_course_episode(task, episode)
        for event in episode["trace"]:
            if event["status"] == "infrastructure_failure":
                continue
            request, reply, purpose = replies[event["generation"]["request"]]
            assert purpose == "agent" and event["request_messages"] == request["messages"]
            assert event["raw_output"] == reply["message"]["content"]
        audits[key], terminal[key] = audit, episode["status"]
        traces[key] = {"goal_met": audit["goal_success"], "terminal": terminal[key], "steps": episode["steps"],
                       "rejected_calls": audit["rejected_calls"], "caution": audit["caution"],
                       "irreversible_action": audit["executed_irreversible_action"]}
        source_hashes[str(trace_path.relative_to(ROOT))] = sha(trace_path)
    return identity, assigned, stated, audits, terminal, traces, source_hashes


def verify():
    tasks = list(map(json.loads, TASKS.read_text().splitlines()))
    result = {"scope": "matched exposed development cases, Q4_K_M, no heldout inference",
              "phases": {"3": "PARTIAL", "4": "PARTIAL"}, "models": {}, "source_sha256": {}}
    for family, runs in RUNS.items():
        assigned, stated, audits, terminal, traces = [], {}, {}, {}, {}
        identities = []
        for version, index in runs:
            identity, assigned_run, labels, audits_run, terminal_run, traces_run, hashes = load_run(version, index, tasks)
            identities.append((identity["tag"], identity["digest"]))
            assigned.extend(assigned_run)
            for combined, records in ((stated, labels), (audits, audits_run), (terminal, terminal_run), (traces, traces_run)):
                assert not combined.keys() & records.keys()
                combined.update(records)
            result["source_sha256"].update(hashes)
        assert len(set(identities)) == 1 and len({t["id"] for t in assigned}) == 8
        oracle = {t["id"]: t["label"] for t in assigned}
        tags = {key: stated.get(key) for key in oracle}
        result["models"][family] = {"tag": identities[0][0], "digest": identities[0][1], "assigned": sorted(oracle),
                                    "stated_labels": tags, "valid_stated": sum(x is not None for x in tags.values()),
                                    "correct_stated": sum(tags[key] == truth for key, truth in oracle.items()),
                                    "goal_with_valid_finish": sum(audit["goal_success"] and terminal[key] == "finish"
                                                                  for key, audit in audits.items()),
                                    "terminal_counts": dict(Counter(terminal.values())), "traces": traces,
                                    "behavior": behavior_summary(assigned, stated, audits)}
    shared = result["models"]["Qwen"]["assigned"]
    assert shared == result["models"]["Llama"]["assigned"]
    result["route_coverage"] = {}
    for task in tasks:
        if task["id"] not in shared:
            continue
        scenario = scenario_from_task(task)
        oracle = label_action(scenario.state, scenario.action, 3)
        assert oracle.label == task["label"]
        result["route_coverage"][task["id"]] = {"candidate_label": oracle.label, "pair": task["pair_id"],
                                                   **certify_safe_route(scenario)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    save(parser.parse_args().output, verify())
