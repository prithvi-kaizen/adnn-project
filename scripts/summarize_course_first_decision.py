"""Independently replay the bounded intervention and plot every assigned cell."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.agent.course_metrics import audit_course_episode  # noqa: E402
from src.models.feature_store import sha  # noqa: E402
from src.probes.statistics import bootstrap_mean, paired_binary  # noqa: E402


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.input / "report.json").read_text())
    assert report["status"] == "PASS_EXPLORATORY_FIRST_DECISION_RECORDS"
    assert report["completes_original_full_RQ4"] is False
    assert report["exact_raw_site_binding"]
    cells = records(args.input / "cells.jsonl")
    requests = records(args.input / "generation_requests.jsonl")
    responses = records(args.input / "generation_responses.jsonl")
    assert len(requests) == len(responses) == report["new_generation_calls"] == 106
    assert {r["request_id"] for r in requests} == {r["request_id"] for r in responses}
    response_map = {r["request_id"]: r for r in responses}
    assert len(response_map) == 106
    task_map = {t["id"]: t for t in records(ROOT / "data/course_fast_track/v2/development_tasks.jsonl")}
    ids = report["assigned_ids"]
    assert len(set(ids)) == 6 and len(cells) == 102
    conditions = [("baseline", 0)] + [(d, a) for d in ("concept", "random_811", "random_812", "random_813")
                                             for a in (-2, -1, 1, 2)]
    assert Counter((c["direction"], c["strength"]) for c in cells) == Counter({c: 6 for c in conditions})
    assert len({(c["id"], c["direction"], c["strength"]) for c in cells}) == 102
    used_generation = []
    for cell in cells:
        task, episode = task_map[cell["id"]], cell["episode"]
        assert cell["id"] in ids and episode["max_steps"] == 1 and len(episode["trace"]) == 1
        assert cell["initial_label"] == task["label"]
        assert audit_course_episode(task, episode) == cell["audit"]
        assert not cell["audit"]["infrastructure_failure"] and not cell["audit"]["unknown_action_labels"]
        event = episode["trace"][0]
        if not cell["baseline_reused"]:
            generation = event["generation"]
            assert response_map[generation["request_id"]] == generation
            assert generation["text"] == event["raw_output"]
            used_generation.append(generation["request_id"])
    assert len(set(used_generation)) == 96
    zero_ids = [r["request_id"] for r in report["zero_validations"]]
    assert len(set(zero_ids)) == 6 and all(r["exact_token_identity"] for r in report["zero_validations"])
    timing_ids = [r["request_id"] for r in report["timing_requests"]]
    assert len(set(timing_ids)) == 4
    assert set(used_generation + zero_ids + timing_ids) == set(response_map)
    assert len(used_generation + zero_ids + timing_ids) == 106
    vectors = np.load(args.input / "directions.npz", allow_pickle=False)
    norms = {key: float(np.linalg.norm(vectors[key].astype(np.float64))) for key in vectors.files}
    assert set(norms) == {"concept", "random_811", "random_812", "random_813"}
    assert all(np.isclose(norm, norms["concept"], rtol=1e-6) for norm in norms.values())
    baseline = {c["id"]: c for c in cells if c["direction"] == "baseline"}

    def endpoints(cell):
        audit = cell["audit"]
        return {"caution": int(audit["caution"]), "goal": int(audit["goal_success"]),
                "legal": int(cell["episode"]["trace"][0]["status"] in {"executed", "terminal"}),
                "any_I": int(audit["executed_irreversible_action"]),
                "shortcut": int(audit["shortcut_executed"])}

    points = []
    for direction, strength in conditions:
        group = sorted((c for c in cells if (c["direction"], c["strength"]) == (direction, strength)), key=lambda c: c["id"])
        assert {c["id"] for c in group} == set(ids)
        point = {"direction": direction, "strength": strength, "N": 6, "template_clusters": 3,
                 "counts": {}, "rates": {}, "paired_difference_from_zero": {}, "statuses": dict(Counter(c["episode"]["trace"][0]["status"] for c in group))}
        for endpoint in endpoints(group[0]):
            values = [endpoints(c)[endpoint] for c in group]
            zeros = [endpoints(baseline[c["id"]])[endpoint] for c in group]
            point["counts"][endpoint] = sum(values)
            point["rates"][endpoint] = bootstrap_mean(values, [c["template_id"] for c in group])
            point["paired_difference_from_zero"][endpoint] = paired_binary(values, zeros, [c["template_id"] for c in group])
        points.append(point)
    summary = {"status": "PASS_INDEPENDENT_FIRST_DECISION_REPLAY", "scope": report["scope"],
               "complete_full_RQ4": False, "first_decision_cells": 102, "fresh_generation_requests": 106,
               "fresh_treatment_cells": 96, "reused_baseline_cells": 6, "zero_identity_checks": 6,
               "timing_calls": 4, "block": report["block"], "direction_norms": norms,
               "elapsed_seconds": report["elapsed_seconds"], "peak_process_memory": report["memory"],
               "points": points, "CI_policy": "1000 joint template draws, seed 9001, conditional on three exposed templates; exploratory, no multiplicity-adjusted claim",
               "human_judgments": 0,
               "source_sha256": {str(p): sha(p) for p in [args.input / "report.json", args.input / "cells.jsonl", Path(__file__)]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2)+"\n")

    sys.path.insert(0, str(ROOT / ".cache/course_fast_track/plotting"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/course_fast_track/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for axis, endpoint, title in zip(axes, ("caution", "goal", "legal"), ("First-decision caution", "Goal after one decision", "Legal first decision")):
        concept = sorted([p for p in points if p["direction"] == "concept"] + [points[0]], key=lambda p: p["strength"])
        axis.fill_between([p["strength"] for p in concept], [p["rates"][endpoint]["CI95"][0] for p in concept],
                          [p["rates"][endpoint]["CI95"][1] for p in concept], color="#17364b", alpha=.1)
        for direction, color, style in [("concept", "#17364b", "-"), ("random_811", "#ad4b29", "--"),
                                         ("random_812", "#276b57", ":"), ("random_813", "#805b9e", "-.")]:
            selected = sorted([p for p in points if p["direction"] == direction] + [points[0]], key=lambda p: p["strength"])
            axis.plot([p["strength"] for p in selected], [p["counts"][endpoint]/6 for p in selected],
                      label=direction.replace("_", " "), color=color, linestyle=style, marker="o", markersize=4)
        axis.set(title=title, xlabel="Signed strength", xticks=[-2, -1, 0, 1, 2], ylim=(-.04, 1.04))
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Fraction of six assigned versions")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", frameon=False, bbox_to_anchor=(.5, .025))
    fig.suptitle("Exploratory one-decision diagnostic, block "+str(report["block"])+"; six versions, three templates", fontsize=12)
    fig.tight_layout(rect=(0, .13, 1, .94))
    for ext in ("pdf", "svg", "png"):
        fig.savefig(args.figures / f"first_decision_steering.{ext}", dpi=180)
    plt.close(fig)
    manifest = {"summary_sha256": sha(args.output), "program_sha256": sha(Path(__file__)),
                "files_sha256": {p.name: sha(p) for p in args.figures.glob("first_decision_steering.*")},
                "visual_QA": "PENDING"}
    (args.figures / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"points", "source_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
