"""Verify the private Mac RQ5 trace and publish only aggregate statistics."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np
from run_bytesized_rq5_mac import PLAN, SOURCE
from run_bytesized_rq5_native import MAX_NEW, ROOT, read_cases, serialize, sha, strict_prediction


def auc(rows, field, *, within):
    groups = sorted({row["game"] for row in rows})
    index = {game: i for i, game in enumerate(groups)}
    wins = np.zeros((len(groups), len(groups)))
    counts = np.zeros_like(wins)
    covered = set()
    positive = [row for row in rows if row["label"] == "irreversible"]
    negative = [row for row in rows if row["label"] == "reversible"]
    for pos in positive:
        for neg in negative:
            if within and (pos["game"], pos["verb"]) != (neg["game"], neg["verb"]):
                continue
            coordinate = index[pos["game"]], index[neg["game"]]
            wins[coordinate] += (pos[field] > neg[field]) + 0.5 * (pos[field] == neg[field])
            counts[coordinate] += 1
            covered.update((pos["id"], neg["id"]))
    games = [i for i in range(len(groups)) if counts[i, :].sum() or counts[:, i].sum()]
    draws = []
    if games:
        rng = np.random.default_rng(9001)
        for selection in rng.choice(games, size=(1000, len(games)), replace=True):
            multiplicity = np.bincount(selection, minlength=len(groups))
            weights = np.diag(multiplicity) if within else np.outer(multiplicity, multiplicity)
            n = float((counts * weights).sum())
            if n:
                draws.append(float((wins * weights).sum()) / n)
    if len(games) >= 3 and len(draws) >= 200:
        interval = np.quantile(draws, [0.025, 0.975]).tolist()
    else:
        interval = None
    total = int(counts.sum())
    return {"auroc": float(wins.sum()) / total if total else None, "CI95": interval,
            "games": [groups[i] for i in games], "scored_case_count": len(covered), "compared_pairs": total,
            "defined_bootstrap_draws": len(draws), "undefined_bootstrap_draws": 1000 - len(draws)}


def analyze(cases_path, run, guard):
    spec = json.loads(SOURCE.read_text())
    cases = read_cases(cases_path, spec["private_design_sha256"])
    records = [json.loads(line) for line in run.read_text().splitlines()]
    lock, preflight = records[:2]
    assert lock["type"] == "lock" and preflight["type"] == "preflight"
    assert lock["protocol_sha256"] == sha(PLAN) and lock["cohort_sha256"] == sha(cases_path)
    assert lock["code_sha256"] == sha(ROOT / "scripts/run_bytesized_rq5_mac.py")
    assert lock["scorer_sha256"] == sha(ROOT / "src/models/continuation_scoring.py")
    assert lock["assigned"] == len(cases) == 82 and len(records) == 2 + 2 * len(cases)
    assert preflight["seconds_with_margin"] <= 1200 and preflight["raw_score_max_error"] <= 1e-4
    assert json.loads((guard / "process.json").read_text())["status"] == "PASS"
    samples = [json.loads(line) for line in (guard / "memory.jsonl").read_text().splitlines()]
    assert samples
    rows = []
    absent = Counter()
    counts = Counter()
    by_label = Counter()
    for idx, case in enumerate(cases):
        request, score = records[2 + 2 * idx:4 + 2 * idx]
        assert request["type"] == "request" and score["type"] == "score" and score["id"] == case["id"]
        assert all(request[key] == val for key, val in case.items())
        assert 0 < len(score["ids"]) <= MAX_NEW and len(score["ids"]) == len(score["nll"]) + score["terminal_eos"]
        assert score["cap_without_eos"] == (not score["terminal_eos"] and len(score["ids"]) == MAX_NEW)
        assert len(request["input_ids"]) + MAX_NEW <= 2048
        prediction = strict_prediction(score["text"])
        assert score["json"] == (prediction is not None) and score["exact_delta"] == (prediction == serialize(case["gold_delta"]))
        assert all(math.isfinite(val) and val >= 0 for val in score["nll"])
        counts.update(field for field in ("json", "exact_delta", "terminal_eos", "cap_without_eos") if score[field])
        if prediction is not None and set(json.loads(prediction)) == set(case["gold_delta"]):
            counts["json_with_required_keys"] += 1
        by_label[case["label"]] += 1
        if score["nll"]:
            rows.append({"id": case["id"], "game": case["game"], "verb": case["verb"], "label": case["label"],
                         "mean_nll": mean(score["nll"]), "content_tokens": len(score["nll"]),
                         "changed_objects": case["changed_object_count"]})
        else:
            absent[case["label"]] += 1
    assert by_label == {"reversible": 37, "irreversible": 45}
    results = {}
    for key in ("mean_nll", "content_tokens", "changed_objects"):
        results[key] = {"aggregate": auc(rows, key, within=False), "within_game_and_verb": auc(rows, key, within=True)}
    label_nll = {label: {"N": len(values), "mean": mean(values) if values else None,
                        "median": median(values) if values else None} for label in by_label
                 for values in [[row["mean_nll"] for row in rows if row["label"] == label]]}
    return {"status": "PASS_LOCAL_RQ5_DIAGNOSTIC_RECOUNT", "scope": lock["scope"], "model": lock["model"],
            "revision": lock["revision"], "device": lock["device"], "dtype": lock["dtype"],
            "assigned": dict(by_label), "scored_cases": len(rows), "absent_scores": dict(absent),
            "output_counts": {field: counts[field] for field in
                              ("json", "json_with_required_keys", "exact_delta", "terminal_eos", "cap_without_eos")},
            "nll_summary_by_label": label_nll, "metrics_irreversible_positive": results,
            "preflight": preflight, "seconds_at_last_guard_reading": max(s["elapsed_seconds"] for s in samples),
            "min_memory_free_percent": min(s["memory_free_percentage"] for s in samples),
            "maximum_added_swap_bytes": max(s["added_swap_bytes"] for s in samples),
            "peak_process_rss_bytes": max(r["process_lifetime_peak_rss_bytes"] for r in records[3::2]),
            "provenance": {"raw_records_sha256": sha(run), "private_cohort_sha256": sha(cases_path),
                           "guard_log_sha256": sha(guard / "memory.jsonl"), "code_sha256": sha(Path(__file__)),
                           "protocol_sha256": sha(PLAN), "lock": lock},
            "original_bf16_bonus_complete": False, "redistributed_upstream_rows": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("guard", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x") as output:
        output.write(json.dumps(analyze(args.cases, args.run, args.guard), indent=2) + "\n")
