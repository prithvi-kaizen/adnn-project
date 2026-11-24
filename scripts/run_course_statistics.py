"""Generate course statistics only after a verified reduced C4 gate."""

import argparse
import gzip
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.models.feature_store import sha  # noqa: E402
from src.probes.course import eligible_indices  # noqa: E402
from src.probes.statistics import (  # noqa: E402
    bootstrap_classification,
    bootstrap_kappa,
    bootstrap_mean,
    execution_rates,
    group_weights,
    interval,
    paired_binary,
)


def filter_record(record, keep):
    return {key: [values[i] for i in keep] if key in ("ids", "truth", "predictions", "probabilities") else values
            for key, values in record.items() if key != "metrics"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--length", type=Path, required=True)
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Never overwrite statistical outputs")
    gate = json.loads(args.gate.read_text())
    assert gate["status"] == "PASS_C4_REDUCED_SCOPE"
    assert gate["summary_sha256"] == sha(args.summary)
    started = time.monotonic()
    summary = json.loads(args.summary.read_text())
    length = json.loads((args.length / "test_predictions.json").read_text())
    length_lock = json.loads((args.length / "selection_lock.json").read_text())
    assert length["selection_lock_sha256"] == sha(args.length / "selection_lock.json")
    length_by_condition = {(r["seed"], r["binary"]): r for r in length["records"]}
    equal_length_pairs = {r["pair_id"] for r in length_lock["pair_tokenization_audit"] if r["prompt_token_difference"] == 0}
    rows = [json.loads(line) for line in (ROOT / "data/course_fast_track/v2/sandbox_pool.jsonl").read_text().splitlines()]
    by_id = {r["id"]: r for r in rows}
    manifests = {s: json.loads((ROOT / f"data/course_fast_track/v2/splits/pool_1800/seed_{s}.json").read_text())
                 for s in (1729, 2718, 31415)}
    with gzip.open(args.probes / "test_predictions.jsonl.gz", "rt") as handle:
        experiments = [json.loads(line) for line in handle]
    experiment_by_id = {r["id"]: r for r in experiments}
    conditions = []
    for count, result in enumerate(experiments, 1):
        c = result["condition"]
        manifest = manifests[c["seed"]]
        layers = result["layers"]
        if not layers:
            conditions.append({"id": result["id"], "condition": c, "status": result["status"]})
            continue
        selected = layers[str(result["selected_layer"])]
        groups = [manifest["components"][key] for key in selected["ids"]]
        order = ("reversible", "irreversible") if c["binary"] else ("reversible", "partially_reversible", "irreversible")
        measured = {"id": result["id"], "condition": c, "status": result["status"],
                    "selected_layer": result["selected_layer"],
                    "selected": bootstrap_classification(selected, groups, order,
                        reference=result.get("lexical") if c["control"] == "ordinary" else None),
                    "layer_accuracy": {}}
        weights, design = group_weights(groups)
        layer_correct = np.array([[a == b for a, b in zip(record["truth"], record["predictions"])]
                                  for record in layers.values()], dtype=float)
        distributions = weights @ layer_correct.T / weights.sum(1)[:, None]
        for j, (layer, record) in enumerate(layers.items()):
            measured["layer_accuracy"][layer] = {"accuracy": record["metrics"]["accuracy"],
                "N": len(record["ids"]), "clusters": design["clusters"],
                **interval(distributions[:, j], design["clusters"])}
        if c["control"] == "ordinary":
            measured["lexical"] = bootstrap_classification(result["lexical"], groups, order)
            measured["majority"] = bootstrap_classification(result["majority"], groups, order)
            if c["horizon"] == 3 and c["subset"] == "all":
                length_reference = length_by_condition[(c["seed"], c["binary"])]
                measured["exploratory_probe_minus_length_control"] = bootstrap_classification(
                    selected, groups, order, reference=length_reference)
                prefix = result["id"].rsplit("_ordinary", 1)[0]
                control = experiment_by_id[prefix + "_verb_777"]
                if control["layers"]:
                    verb = control["layers"][str(control["selected_layer"])]
                    assert selected["ids"] == verb["ids"]
                    difference = [int(a == y) - int(b == z) for a, y, b, z in zip(selected["predictions"],
                        selected["truth"], verb["predictions"], verb["truth"])]
                    measured["selectivity_accuracy_difference"] = bootstrap_mean(difference, groups)
                else:
                    measured["selectivity_accuracy_difference"] = {"status": control["status"], "CI95": None}
                shuffled = [experiment_by_id[prefix + f"_shuffle_{s}"] for s in range(101, 111)]
                shuffled_records = [r["layers"][str(r["selected_layer"])] for r in shuffled]
                assert all(r["ids"] == selected["ids"] and r["truth"] == selected["truth"] for r in shuffled_records)
                correctness = np.array([[a == b for a, b in zip(r["truth"], r["predictions"])] for r in shuffled_records])
                measured["mean_10_shuffle_accuracy"] = bootstrap_mean(correctness.mean(0), groups)
                measured["mean_10_shuffle_accuracy"]["conditional_on_realized_training_permutations"] = True
            if c["horizon"] == 3 and c["binary"] and c["subset"] == "all":
                eligible = {rows[i]["id"] for i in eligible_indices(rows, 3, True, True)}
                indices = [i for i, key in enumerate(selected["ids"]) if key in eligible]
                pairs = {}
                for i in indices:
                    key = by_id[selected["ids"][i]]["minimal_pair_id"]
                    pairs.setdefault(key, []).append(i)
                assert all(len(values) == 2 for values in pairs.values())
                pair_correct, shifts, pair_groups = [], [], []
                positive = selected["classes"].index("irreversible")
                for members in pairs.values():
                    i = next(j for j in members if selected["truth"][j] == "irreversible")
                    r = next(j for j in members if selected["truth"][j] == "reversible")
                    pair_correct.append(int(all(selected["truth"][j] == selected["predictions"][j] for j in members)))
                    shifts.append(selected["probabilities"][i][positive] - selected["probabilities"][r][positive])
                    assert groups[i] == groups[r]
                    pair_groups.append(groups[i])
                measured["heldout_minimal_pairs"] = {
                    "N_pairs": len(pairs), "N_members": len(indices),
                    "primary_probe_minus_lexical": bootstrap_classification(filter_record(selected, indices),
                        [groups[j] for j in indices], order, filter_record(result["lexical"], indices)),
                    "joint_pair_correctness": bootstrap_mean(pair_correct, pair_groups),
                    "irreversible_probability_shift": bootstrap_mean(shifts, pair_groups)}
                equal_indices = [j for j in indices if by_id[selected["ids"][j]]["minimal_pair_id"] in equal_length_pairs]
                equal_pair_indices = [j for j, key in enumerate(pairs) if key in equal_length_pairs]
                if equal_indices:
                    equal_groups = [groups[j] for j in equal_indices]
                    measured["equal_length_RI_pairs_secondary"] = {
                        "scope": "exploratory_prespecified_before_test_performance",
                        "N_pairs": len(equal_pair_indices), "N_members": len(equal_indices),
                        "probe_minus_lexical": bootstrap_classification(filter_record(selected, equal_indices),
                            equal_groups, order, filter_record(result["lexical"], equal_indices)),
                        "probe_minus_length": bootstrap_classification(filter_record(selected, equal_indices),
                            equal_groups, order, filter_record(length_reference, equal_indices)),
                        "joint_pair_correctness": bootstrap_mean([pair_correct[j] for j in equal_pair_indices],
                            [pair_groups[j] for j in equal_pair_indices])}
                else:
                    measured["equal_length_RI_pairs_secondary"] = {"status": "NOT_ESTIMABLE_NO_ELIGIBLE_PAIRS", "N_pairs": 0}
        conditions.append(measured)
        if count % 10 == 0:
            print(f"statistics: {count}/180 conditions", flush=True)
    agreements = []
    for record in summary["RQ2"]:
        joined = record["joined"]
        manifest = manifests[record["seed"]]
        groups = [manifest["components"][r["id"]] for r in joined]
        stated = {"ids": [r["id"] for r in joined], "truth": [r["truth"] for r in joined],
                  "predictions": [r["stated"] for r in joined]}
        valid = [i for i, row in enumerate(joined) if row["stated"] != "invalid"]
        agreements.append({"seed": record["seed"], "pooling": record["pooling"],
            "stated_all_ID": bootstrap_classification(stated, groups,
                ("reversible", "partially_reversible", "irreversible")),
            "probe_stated_kappa_all_ID": bootstrap_kappa([r["probe"] for r in joined], [r["stated"] for r in joined], groups),
            "probe_stated_kappa_valid_only": bootstrap_kappa([joined[i]["probe"] for i in valid],
                [joined[i]["stated"] for i in valid], [groups[i] for i in valid]) if valid else None,
            "valid_count": len(valid), "assigned_N": len(joined)})
    pairs = summary["RQ3"]["pairs"]
    behavior = {"assigned_RI_pairs": len(pairs), "template_clusters": len({p["template_id"] for p in pairs}),
                "Gap_bounds": summary["RQ3"]["Gap_bounds"], "point_and_CI": None,
                "nominal_McNemar_ignores_within_template_dependence": True}
    behavior_report = json.loads((args.behavior / "report.json").read_text())
    audits = [r["audit"] for r in behavior_report["episodes"]]
    by_task = {a["id"]: a for a in audits}
    behavior["execution_rates_all_assigned_versions"] = execution_rates(audits, 24)
    behavior["selected_terminal_status_counts"] = dict(Counter(r["status"] for r in behavior_report["episodes"]))
    if all(p["fully_observed"] for p in pairs):
        groups = [p["template_id"] for p in pairs]
        behavior["point_and_CI"] = {key: bootstrap_mean([p[key] for p in pairs], groups)
                                    for key in ("K", "A", "gap", "sensitivity", "C_R", "C_I")}
        behavior["nominal_paired_caution"] = paired_binary([p["C_I"] for p in pairs], [p["C_R"] for p in pairs], groups)
        behavior["paired_endpoints"] = {}
        for name in ("goal_success", "shortcut_executed", "executed_irreversible_action"):
            selected_audits = [by_task[p[key]] for p in pairs for key in ("I_id", "R_id")]
            if name == "goal_success":
                unavailable = any(a["infrastructure_failure"] for a in selected_audits)
            elif name == "shortcut_executed":
                unavailable = any(a["infrastructure_failure"] and not a[name] for a in selected_audits)
            else:
                unavailable = any((a["infrastructure_failure"] or a["unknown_action_labels"]) and not a[name]
                                  for a in selected_audits)
            if unavailable:
                behavior["paired_endpoints"][name] = {"status": "NOT_ESTIMABLE_UNOBSERVED_OUTCOME", "CI95": None}
                continue
            i = [int(by_task[p["I_id"]][name]) for p in pairs]
            r = [int(by_task[p["R_id"]][name]) for p in pairs]
            behavior["paired_endpoints"][name] = {"I": bootstrap_mean(i, groups), "R": bootstrap_mean(r, groups),
                                                 "paired": paired_binary(i, r, groups)}
    value = {"status": "STATISTICS_COMPUTED_PENDING_INDEPENDENT_VERIFICATION_AND_FIGURES",
             "source_sha256": {str(path): sha(path) for path in (args.summary, args.gate,
                 args.probes / "test_predictions.jsonl.gz", ROOT / "src/probes/statistics.py",
                 ROOT / "docs/COURSE_STATISTICS_PROTOCOL.md", Path(__file__))},
             "conditions": conditions, "RQ2": agreements, "RQ3": behavior,
             "bootstrap_seed": 9001, "bootstrap_draws": 1000,
             "two_model_Holm_comparison": "NOT_COMPLETED_ONLY_ONE_NATIVE_FAMILY",
             "elapsed_seconds": time.monotonic() - started}
    value["source_sha256"][str(args.length / "test_predictions.json")] = sha(args.length / "test_predictions.json")
    value["source_sha256"][str(args.behavior / "report.json")] = sha(args.behavior / "report.json")
    value["exploratory_length_control"] = []
    for record in length["records"]:
        manifest = manifests[record["seed"]]
        groups = [manifest["components"][key] for key in record["ids"]]
        order = ("reversible", "irreversible") if record["binary"] else ("reversible", "partially_reversible", "irreversible")
        value["exploratory_length_control"].append({"id": record["id"], "seed": record["seed"],
            "binary": record["binary"], "statistics": bootstrap_classification(record, groups, order)})
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print("Computed saved-data cluster statistics; independent verification and figures still required")


if __name__ == "__main__":
    main()
