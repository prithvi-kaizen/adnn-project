"""Verify saved intervals with explicit replicated rows and independent sklearn metrics."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from run_course_layerwise import EXECUTION, PREDICTIONS, ROOT, binary_order, digest, read, same
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from verify_course_statistics import check_interval


def explicit_rows(groups):
    names = sorted(set(groups))
    members = [np.flatnonzero(np.asarray(groups) == name) for name in names]
    draws = np.random.default_rng(9001).integers(len(names), size=(1000, len(names)))
    unique_draws, inverse = np.unique(np.sort(draws, axis=1), axis=0, return_inverse=True)
    return [np.concatenate([members[index] for index in draw]) for draw in unique_draws], inverse


def explicit_metrics(record, labels, indices):
    truth = np.asarray(record["truth"])[indices]
    predictions = np.asarray(record["predictions"])[indices]
    complete = set(truth) == set(labels)
    auc = None
    if complete and set(record["classes"]) == set(labels):
        probabilities = np.asarray(record["probabilities"])[indices]
        if len(labels) == 2:
            auc = roc_auc_score(truth == "irreversible", probabilities[:, record["classes"].index("irreversible")])
        else:
            order = np.argsort(record["classes"])
            auc = roc_auc_score(truth, probabilities[:, order], labels=np.asarray(record["classes"])[order],
                                multi_class="ovr", average="macro")
    return {"accuracy": accuracy_score(truth, predictions),
            "balanced_accuracy": balanced_accuracy_score(truth, predictions) if complete else None,
            "macro_F1": f1_score(truth, predictions, labels=labels, average="macro", zero_division=0),
            "AUROC": auc}


def verify(path):
    with gzip.open(path, "rt") as handle:
        observed = json.load(handle)
    assert observed["bootstrap_draws"] == 1000 and observed["bootstrap_seed"] == 9001
    assert observed["new_model_calls"] == observed["probe_refits"] == observed["human_labels_created"] == 0
    assert all(digest(Path(source)) == checksum for source, checksum in observed["source_sha256"].items())
    with gzip.open(ROOT / PREDICTIONS, "rt") as handle:
        raw = {row["id"]: row for row in map(json.loads, handle)}
    manifests = {int(seed): read(item["path"]) for seed, item in read(EXECUTION)["split_manifests"].items()}
    assert list(raw) == [row["id"] for row in observed["conditions"]]
    checked, estimable, conditions = Counter(), Counter(), Counter()
    explicit_checks, selectivity_checks, missing_draw_cells = 0, 0, 0
    for result in observed["conditions"]:
        source = raw[result["id"]]
        condition = source["condition"]
        assert result["condition"] == condition
        assert result["locked_selected_layer"] == source["selected_layer"]
        assert result["layers"].keys() == source["layers"].keys()
        ids = source["layers"]["0"]["ids"]
        groups = [manifests[condition["seed"]]["components"][key] for key in ids]
        clusters = len(set(groups))
        assert result["N"] == len(ids) and result["design"]["clusters"] == clusters
        labels = binary_order(condition)
        control = raw.get(result["id"].removesuffix("_ordinary") + "_verb_777")
        sample = (condition["horizon"] == 3 and condition["subset"] == "all" and
                  condition["control"] in ("ordinary", "verb_777", "shuffle_101"))
        if sample:
            indices, inverse = explicit_rows(groups)
        conditions[result["selectivity_status"]] += 1
        for layer, item in result["layers"].items():
            record = source["layers"][layer]
            assert record["ids"] == ids
            missing_draw_cells += item["missing_true_class_draws"] > 0
            assert len(item["metrics"]) == 4
            for metric, data in item["metrics"].items():
                same(data["estimate"], record["metrics"][metric])
                assert data["defined_draws"] + data["undefined_draws"] == 1000
                assert (data["CI95"] is not None) == (clusters >= 3 and data["defined_draws"] >= 200)
                checked[metric] += 1
                estimable[metric] += data["CI95"] is not None
            if sample and int(layer) in (0, source["selected_layer"]):
                measured = [explicit_metrics(record, labels, ix) for ix in indices]
                for metric, data in item["metrics"].items():
                    check_interval(data, [measured[index][metric] for index in inverse], clusters)
                    explicit_checks += 1
            if control:
                verb = control["layers"][layer]
                assert verb["ids"] == ids
                first = np.asarray(record["truth"]) == record["predictions"]
                second = np.asarray(verb["truth"]) == verb["predictions"]
                difference = first.astype(int) - second
                check_interval(item["selectivity"], np.asarray([difference[ix].mean() for ix in indices])[inverse],
                               clusters)
                same(item["selectivity"]["estimate"], difference.mean())
                selectivity_checks += 1
                estimable["selectivity"] += item["selectivity"]["CI95"] is not None
            else:
                assert "selectivity" not in item
    assert len(raw) == 180 and set(checked.values()) == {4500}
    assert selectivity_checks == 300
    return {"status": "PASS_SAVED_RECORD_LAYERWISE_SUPPLEMENT", "statistics_sha256": digest(path.resolve()),
            "verification_source_sha256": {source: digest(Path(source)) for source in (
                "scripts/verify_course_layerwise.py", "scripts/verify_course_statistics.py")},
            "conditions": len(raw), "metrics_checked_against_saved_points": dict(checked),
            "classification_intervals_recomputed_with_explicit_rows": explicit_checks,
            "selectivity_intervals_recomputed_with_explicit_rows": selectivity_checks,
            "conditions_by_selectivity_status": dict(conditions),
            "estimable_intervals_by_metric": dict(estimable),
            "layer_cells_with_missing_true_class_draws": missing_draw_cells,
            "complete_improvement_I5_gate": False, "complete_original_R079": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("Choose a new receipt; previous results are immutable")
    result = verify(args.statistics)
    with args.receipt.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
