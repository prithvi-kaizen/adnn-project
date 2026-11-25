"""Extend the existing full-episode steering grid with seed 812 and 813 controls."""

import argparse
import os
import time
from pathlib import Path

from run_bytesized_rq5_native import sha
from run_course_full_episode_diagnostic import (
    EXECUTION,
    FIRST,
    ROOT,
    audit_course_episode,
    original_episodes,
    read,
    read_rows,
    run_episode,
    save,
    scenario_from_task,
    tasks,
    token_trace,
    verify,
)

PRIOR = ROOT / "artifacts/course_fast_track/native_full_episode_steering_v1"
PLAN = ROOT / "docs/PRESENTATION_STEERING_CONTROLS_PROTOCOL.md"
ADDITIONAL = [(f"random_{seed}", strength) for seed in (812, 813) for strength in (-2, -1, 1, 2)]


def summarize(directory):
    original = original_episodes()
    task_map = tasks()
    previous = verify(PRIOR)
    assert previous == read(PRIOR / "verification.json") and previous["cells"] == 54
    lock = read(directory / "lock.json")
    for path, digest in lock["sha256"].items():
        assert sha(ROOT / path) == digest
    cells = read_rows(directory / "cells.jsonl")
    expected = [(direction, strength, task) for direction, strength in ADDITIONAL for task in task_map]
    assert [(cell["direction"], cell["strength"], cell["id"]) for cell in cells] == expected
    requests = read_rows(directory / "generation_requests.jsonl")
    responses = read_rows(directory / "generation_responses.jsonl")
    turns = [event for row in cells for event in row["episode"]["trace"]]
    assert len(requests) == len(responses) == len(turns)
    for idx, (request, response, event) in enumerate(zip(requests, responses, turns), 1):
        assert request["request_id"] == response["request_id"] == idx
        assert request["messages"] == event["request_messages"] and response == event["generation"]
        assert not request["do_sample"] and request["use_cache"] and request["max_new_tokens"] == 128
    combined = list(previous["conditions"])
    for idx, (direction, strength) in enumerate(ADDITIONAL):
        values = {key: 0 for key in ("caution", "goal", "goal_with_finish", "irreversible", "rejected_calls",
                                    "invalid_abort", "token_trace_changed", "world_trace_changed")}
        pairs = {}
        for row in cells[idx * 6:(idx + 1) * 6]:
            task, episode = task_map[row["id"]], row["episode"]
            audit = audit_course_episode(task, episode)
            assert audit == row["audit"] and not audit["infrastructure_failure"]
            baseline = original[row["id"]]
            values["caution"] += audit["caution"]
            values["goal"] += audit["goal_success"]
            values["goal_with_finish"] += audit["goal_success"] and episode["status"] == "finish"
            values["irreversible"] += audit["executed_irreversible_action"]
            values["rejected_calls"] += audit["rejected_calls"]
            values["invalid_abort"] += episode["status"] == "consecutive_invalid_abort"
            values["token_trace_changed"] += token_trace(episode) != token_trace(baseline)
            values["world_trace_changed"] += [e["after_hash"] for e in episode["trace"]] != [e["after_hash"] for e in baseline["trace"]]
            pairs.setdefault(task["pair_id"], {})[task["label"]] = audit["caution"]
        combined.append({"direction": direction, "strength": strength, "N": 6, **values,
                         "paired_caution_I_gt_R": sum(p["irreversible"] > p["reversible"] for p in pairs.values()),
                         "N_pairs": len(pairs)})
    assert len(combined) == 17 and sum(x["N"] for x in combined) == 102
    return {"status": "PASS_FULL_EPISODE_DEVELOPMENT_CONTROL_GRID", "previous_episodes": 54,
            "new_episodes": len(cells), "total_episodes": 102, "new_model_turns": len(turns),
            "total_model_turns": previous["turns"] + len(turns), "shared_zero_replays": 6,
            "new_records_sha256": sha(directory / "cells.jsonl"), "conditions": combined,
            "original_held_out_rq4_complete": False}


def run(directory):
    os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import numpy as np
    import torch

    from scripts.stage_course_checkpoint import file_receipt
    from src.models.course_native import NativeCourseRunner, append_record
    from src.models.native_smoke import steer_last_position
    from src.models.ollama_dev import api

    if api("/api/ps", timeout=5).get("models", []):
        raise RuntimeError("An Ollama model is already resident")
    execution = read(EXECUTION)
    settings = ROOT / execution["pilot_settings"]
    cache = Path(read(ROOT / execution["checkpoint_receipt"])["snapshot"])
    source = ROOT / "docs/evidence/course_phase1_20261115/fallback_model_metadata.json"
    prior = read(PRIOR / "lock.json")
    assert {item["path"]: file_receipt(cache / item["path"], item) for item in read(source)["files"]} == prior["weights"]
    assert verify(PRIOR) == read(PRIOR / "verification.json") and prior["block"] == 13
    vector_path = FIRST / "directions.npz"
    assert sha(vector_path) == read(FIRST / "report.json")["direction_sha256"]
    with np.load(vector_path, allow_pickle=False) as arrays:
        vectors = {name: arrays[name] for name in ("concept", "random_812", "random_813")}
    assert all(np.isclose(np.linalg.norm(v), np.linalg.norm(vectors["concept"]), rtol=1e-6) for v in vectors.values())
    paths = (Path(__file__).resolve(), PLAN, EXECUTION, settings, vector_path, PRIOR / "verification.json", source)
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / "lock.json", {"sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
                                   "block": 13, "seed": 1729, "weights": prior["weights"], "torch": torch.__version__})
    torch.manual_seed(1729)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.monotonic()
    runner = NativeCourseRunner(cache, read(settings), directory)
    layer = runner.model.model.layers[12]
    for direction, strength in ADDITIONAL:
        for task in tasks().values():
            hooks = len(layer._forward_hooks)
            with steer_last_position(layer, torch.from_numpy(vectors[direction]), strength):
                episode = run_episode(scenario_from_task(task), runner.generate, max_steps=15)
            assert len(layer._forward_hooks) == hooks
            audit = audit_course_episode(task, episode)
            append_record(directory / "cells.jsonl", {"id": task["id"], "direction": direction, "strength": strength,
                                                       "episode": episode, "audit": audit})
            if audit["infrastructure_failure"]:
                raise RuntimeError(f"Infrastructure failure in {task['id']}")
        print(f"Finished {direction} at strength {strength:+d}", flush=True)
    save(directory / "runtime.json", {"seconds": time.monotonic() - started, **runner.memory()})
    save(directory / "verification.json", summarize(directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify-to", type=Path)
    args = parser.parse_args()
    if args.verify_to:
        save(args.verify_to, summarize(args.directory))
    else:
        run(args.directory)
