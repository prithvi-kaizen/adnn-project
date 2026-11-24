"""Run and replay the scoped full-episode course steering diagnostic."""

import argparse
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.course_metrics import audit_course_episode  # noqa: E402
from src.agent.loop import run_episode, trace_signature  # noqa: E402
from src.data.course import scenario_from_task  # noqa: E402

FIRST = ROOT / "artifacts/course_fast_track/native_first_decision_steering_v1"
BASELINE = ROOT / "artifacts/course_fast_track/native_mac_cpu_v1/behavior"
EXECUTION = ROOT / "configs/course_execution_native_mac_cpu_v1.json"
PROTOCOL = ROOT / "docs/COURSE_FULL_EPISODE_STEERING_DIAGNOSTIC.md"
TASKS = ROOT / "data/course_fast_track/v2/development_tasks.jsonl"
PAIRS = {"pilot-delete-0", "pilot-mail-0", "pilot-transfer-0"}
CONDITIONS = [("zero", 0)] + [
    (direction, strength)
    for direction in ("concept", "random_811") for strength in (-2, -1, 1, 2)
]


def read(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def tasks():
    selected = {row["id"]: row for row in read_rows(TASKS) if row["pair_id"] in PAIRS}
    assert len(selected) == 6 and all(row["RI_gap_eligible"] for row in selected.values())
    return selected


def original_episodes():
    index = {row["id"]: row["episode_file"] for row in read(BASELINE / "report.json")["episodes"]}
    return {key: read(BASELINE / index[key]) for key in tasks()}


def token_trace(episode):
    return [event["generation"]["output_token_ids"] for event in episode["trace"]]


def verify(directory):
    from hashlib import sha256

    lock = read(directory / "lock.json")
    for name, expected in lock["source_sha256"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == expected
    task_map, originals = tasks(), original_episodes()
    cells = read_rows(directory / "cells.jsonl")
    assert [(cell["direction"], cell["strength"], cell["id"]) for cell in cells] == [
        (direction, strength, task_id) for direction, strength in CONDITIONS for task_id in task_map
    ]
    requests = read_rows(directory / "generation_requests.jsonl")
    responses = read_rows(directory / "generation_responses.jsonl")
    raw_turns = [event for cell in cells for event in cell["episode"]["trace"]]
    assert len(requests) == len(responses) == len(raw_turns)
    for idx, (request, response, event) in enumerate(zip(requests, responses, raw_turns), 1):
        assert request["request_id"] == response["request_id"] == idx
        assert event["request_messages"] == request["messages"] and event["generation"] == response
        assert response["text"] == event["raw_output"] and not request["do_sample"] and request["use_cache"]
        assert request["max_new_tokens"] == 128 and request["min_new_tokens"] == 0
        assert request["prompt_sha256"] == sha256(request["rendered_prompt"].encode()).hexdigest()
    totals = []
    for direction, strength in CONDITIONS:
        group = [cell for cell in cells if (cell["direction"], cell["strength"]) == (direction, strength)]
        effects = {"caution": 0, "goal": 0, "goal_with_finish": 0, "irreversible": 0, "rejected_calls": 0,
                   "invalid_abort": 0, "token_trace_changed": 0, "world_trace_changed": 0}
        by_pair = {}
        for cell in group:
            task, ep = task_map[cell["id"]], cell["episode"]
            assert ep["max_steps"] == 15 and ep["scenario_id"] == cell["id"]
            audit = audit_course_episode(task, ep)
            assert audit == cell["audit"] and not audit["infrastructure_failure"]
            original = originals[cell["id"]]
            changed = token_trace(ep) != token_trace(original)
            world_changed = [e["after_hash"] for e in ep["trace"]] != [e["after_hash"] for e in original["trace"]]
            if direction == "zero":
                assert not changed and trace_signature(ep) == trace_signature(original) and ep["status"] == original["status"]
            for name, value in {
                "caution": audit["caution"], "goal": audit["goal_success"],
                "goal_with_finish": audit["goal_success"] and ep["status"] == "finish",
                "irreversible": audit["executed_irreversible_action"], "rejected_calls": audit["rejected_calls"],
                "invalid_abort": ep["status"] == "consecutive_invalid_abort", "token_trace_changed": changed,
                "world_trace_changed": world_changed,
            }.items():
                effects[name] += int(value)
            by_pair.setdefault(task["pair_id"], {})[task["label"]] = audit["caution"]
        paired = sum(p["irreversible"] > p["reversible"] for p in by_pair.values())
        totals.append({"direction": direction, "strength": strength, "N": len(group), **effects,
                       "paired_caution_I_gt_R": paired, "N_pairs": len(by_pair)})
    return {"status": "PASS_EXPLORATORY_FULL_EPISODE_REPLAY", "cells": len(cells), "turns": len(responses),
            "zero_exact_traces": 6, "conditions": totals, "completes_original_full_RQ4": False}


def execute(directory):
    os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTORCH_ENABLE_MPS_FALLBACK="0")
    import numpy as np
    import torch

    from scripts.stage_course_checkpoint import file_receipt
    from src.models.course_native import NativeCourseRunner, append_record
    from src.models.feature_store import sha
    from src.models.native_smoke import steer_last_position
    from src.models.ollama_dev import api

    directory.mkdir(parents=True, exist_ok=False)
    previous = read(FIRST / "report.json")
    assert previous["status"] == "PASS_EXPLORATORY_FIRST_DECISION_RECORDS" and previous["block"] == 13
    vector_path = FIRST / "directions.npz"
    assert sha(vector_path) == previous["direction_sha256"]
    with np.load(vector_path, allow_pickle=False) as arrays:
        vectors = {key: arrays[key] for key in ("concept", "random_811")}
    assert np.isclose(np.linalg.norm(vectors["concept"]), np.linalg.norm(vectors["random_811"]), rtol=1e-6)
    execution = read(EXECUTION)
    assert previous["execution"] == execution
    settings = ROOT / execution["pilot_settings"]
    assert (execution["device"], execution["precision"], execution["cpu_threads"]) == ("cpu", "float32", 4)
    runner_path = ROOT / "src/models/course_native.py"
    assert sha(runner_path) == execution["model_runner_source_sha256"]
    versions = {p: importlib.metadata.version(p) for p in previous["packages"]}
    assert versions == previous["packages"]
    cache = Path(read(ROOT / execution["checkpoint_receipt"])["snapshot"])
    cache_spec = read(ROOT / "docs/evidence/course_phase1_20261115/fallback_model_metadata.json")
    assert {spec["path"]: file_receipt(cache / spec["path"], spec) for spec in cache_spec["files"]} == previous["checkpoint_files"]
    if api("/api/ps", timeout=5).get("models", []):
        raise RuntimeError("Ollama already has a resident model")
    task_map, originals = tasks(), original_episodes()
    locked = (Path(__file__), PROTOCOL, EXECUTION, settings, TASKS, FIRST / "report.json", vector_path,
              BASELINE / "report.json", runner_path, ROOT / "src/models/native_smoke.py", ROOT / "src/agent/loop.py",
              ROOT / "src/agent/course_metrics.py")
    save(directory / "lock.json", {"source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in locked},
         "conditions": CONDITIONS, "task_ids": list(task_map), "seed": 1729, "packages": versions,
         "hardware": platform.platform(), "block": 13, "weights": previous["checkpoint_files"]})
    random.seed(1729)
    np.random.seed(1729)
    torch.manual_seed(1729)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    start = time.monotonic()
    model = NativeCourseRunner(cache, read(settings), directory, lambda: None)
    layer = model.model.model.layers[12]
    for direction, strength in CONDITIONS:
        vector = torch.from_numpy(vectors["concept" if direction == "zero" else direction])
        for task_id, task in task_map.items():
            count_before = len(layer._forward_hooks)
            with steer_last_position(layer, vector, strength):
                episode = run_episode(scenario_from_task(task), model.generate, max_steps=15)
            assert len(layer._forward_hooks) == count_before
            audit = audit_course_episode(task, episode)
            cell = {"id": task_id, "direction": direction, "strength": strength, "episode": episode, "audit": audit}
            append_record(directory / "cells.jsonl", cell)
            if audit["infrastructure_failure"]:
                raise RuntimeError(f"Infrastructure failure in {direction}/{strength}/{task_id}")
            if direction == "zero" and token_trace(episode) != token_trace(originals[task_id]):
                raise RuntimeError(f"Zero-strength token mismatch in {task_id}")
        print(f"Completed direction {direction} at strength {strength:+d}", flush=True)
    save(directory / "runtime.json", {"seconds": time.monotonic() - start, **model.memory()})
    save(directory / "verification.json", verify(directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify-to", type=Path)
    args = parser.parse_args()
    if args.verify_to:
        save(args.verify_to, verify(args.directory))
    else:
        execute(args.directory)
