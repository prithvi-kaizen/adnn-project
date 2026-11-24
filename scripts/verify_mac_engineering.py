"""Recheck the cached Llama readout and interactive reviewer records offline."""

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from check_mac_reviewer_interactive import review
from check_mac_reviewer_transitions import CONFIG, fixtures
from course_llm_audit import ROOT, load_json, save, sha, witness_result


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def memory(directory):
    samples = lines(directory / "memory.jsonl")
    return {"samples": len(samples), "min_memory_free_percentage": min(s["memory_free_percentage"] for s in samples),
            "max_added_swap_bytes": max(s["added_swap_bytes"] for s in samples),
            "elapsed_seconds_at_last_sample": samples[-1]["elapsed_seconds"]}


def layers(directory):
    files = [directory / arm / "pooled_action_vectors.f32" for arm in ("smoke", "repeat")]
    raw = [directory / arm / "stdout.log" for arm in ("smoke", "repeat")]
    assert sha(files[0]) == sha(files[1]) and sha(raw[0]) == sha(raw[1])
    rows = lines(raw[0])
    assert len(rows) == 2 and [r["recycle"] for r in rows] == [True, False]
    for key in ("prompt_tokens", "action_positions", "action_token_ids", "block_inputs", "width"):
        assert rows[0][key] == rows[1][key]
    assert all(r["max_abs_logits_difference"] <= 1e-4 for r in rows)
    shape = (2, 2, rows[0]["block_inputs"], rows[0]["width"])
    features = np.fromfile(files[0], dtype="<f4").reshape(shape)
    assert np.isfinite(features).all() and np.array_equal(features[0, :, 0], features[1, :, 0])
    assert not np.array_equal(features[0, :, -1], features[1, :, -1])
    return {"scope": "technical smoke on one synthetic pair; no fit or holdout",
            "shape": shape, "axis_order": ["case", "last_or_mean", "block_input", "width"],
            "tokens": rows, "byte_identical_repeat": True, "features_sha256": sha(files[0]),
            "first_block_equal_between_versions": True, "last_block_max_absolute_difference":
                float(np.max(np.abs(features[0, :, -1] - features[1, :, -1]))),
            "memory": {arm: memory(directory / arm) for arm in ("build", "smoke", "repeat")}}


def reviewer(directory, model):
    stored = load_json(directory / "report.json")
    assert not stored["resident_after"] and stored["study_cases_sent"] == 0
    assert stored["model"]["digest"] == model["digest"]
    assert len(stored["cases"]) == len(fixtures())
    statuses, completed, responses = Counter(), 0, 0
    truncations = 0
    for i, ((before, action, horizon, expected), reported) in enumerate(zip(fixtures(), stored["cases"]), 1):
        base = directory / f"fixture_{i}"
        count = 0

        def transport(request):
            nonlocal count, truncations
            assert request == load_json(base / f"{count}.request.json")
            raw = (base / f"{count}.response.json").read_bytes()
            count += 1
            truncations += int(json.loads(raw).get("done_reason") == "length")
            return 200, raw

        with tempfile.TemporaryDirectory() as temp:
            recovered = review(Path(temp), model, before, action, horizon, transport)
        assert count == reported["calls"] == len(list(base.glob("*.response.json")))
        assert all(reported[key] == value for key, value in recovered.items())
        witness = witness_result(recovered, {"state_text": before.text(), "action_text": action.text()})
        required = {"reversible": "exact", "partially_reversible": "paid", "irreversible": "not_claimed"}
        passed = recovered["label"] == expected and witness == required[expected]
        assert (expected, witness, passed) == (reported["expected"], reported["witness_validation"], reported["correct"])
        completed += passed
        statuses[recovered["status"]] += 1
        responses += count
    assert completed == stored["passed"]
    return {"assigned": len(fixtures()), "passed": completed, "responses": responses, "truncated_responses": truncations,
            "case_status_counts": dict(statuses), "memory": memory(directory), "report_sha256": sha(directory / "report.json")}


def verify():
    base = ROOT / "artifacts/exploratory"
    config = load_json(CONFIG)
    return {"native_llama": layers(base / "mac_llamacpp_layers_v1"), "reviewers": {
        name: reviewer(base / "mac_reviewer_interactive_v1" / name, config["models"][judge])
        for name, judge in (("qwen", "B"), ("mistral", "A"))}, "study_cases_opened": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    save(parser.parse_args().output, verify())
