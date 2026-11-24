"""Independent explicit-row bootstrap checks for primary saved course results."""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score  # noqa: E402

from src.models.feature_store import sha  # noqa: E402


def check_interval(actual, values, groups):
    finite = np.array([x for x in values if x is not None and np.isfinite(x)])
    assert actual["defined_draws"] == len(finite) and actual["undefined_draws"] == 1000-len(finite)
    if groups >= 3 and len(finite) >= 200:
        assert np.allclose(actual["CI95"], np.percentile(finite, [2.5, 97.5]), rtol=0, atol=1e-12)
    else:
        assert actual["CI95"] is None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new independent verification receipt")
    stats = json.loads(args.statistics.read_text())
    summary = json.loads(args.summary.read_text())
    assert all(sha(ROOT / path if not Path(path).is_absolute() else Path(path)) == value
               for path, value in stats["source_sha256"].items())
    with gzip.open(args.probes / "test_predictions.jsonl.gz", "rt") as handle:
        raw = {r["id"]: r for r in map(json.loads, handle)}
    independent_primary_checks = 0
    point_checks = 0
    for condition in stats["conditions"]:
        if condition["status"] != "SELECTED":
            continue
        source = raw[condition["id"]]
        record = source["layers"][str(source["selected_layer"])]
        for key, item in condition["selected"]["metrics"].items():
            expected = record["metrics"][key]
            if expected is None:
                assert item["estimate"] is None
            else:
                assert np.isclose(item["estimate"], expected, rtol=0, atol=1e-12)
            assert item["defined_draws"] + item["undefined_draws"] == 1000
            point_checks += 1
        c = condition["condition"]
        if not (c["horizon"] == 3 and c["pooling"] == "last" and c["subset"] == "all" and c["control"] == "ordinary"):
            continue
        manifest = json.loads((ROOT / f"data/course_fast_track/v2/splits/pool_1800/seed_{c['seed']}.json").read_text())
        groups = [manifest["components"][key] for key in record["ids"]]
        names = sorted(set(groups))
        members = {g: np.array([i for i, key in enumerate(groups) if key == g]) for g in names}
        draws = np.random.default_rng(9001).integers(0, len(names), size=(1000, len(names)))
        y, prediction = np.array(record["truth"]), np.array(record["predictions"])
        p, classes = np.array(record["probabilities"]), np.array(record["classes"])
        order = ["reversible", "irreversible"] if c["binary"] else ["reversible", "partially_reversible", "irreversible"]
        samples = {key: [] for key in ("accuracy", "balanced_accuracy", "macro_F1", "AUROC")}
        for draw in draws:
            ix = np.concatenate([members[names[k]] for k in draw])
            truth, predicted = y[ix], prediction[ix]
            complete = set(truth) == set(order)
            samples["accuracy"].append(float(accuracy_score(truth, predicted)))
            samples["macro_F1"].append(float(f1_score(truth, predicted, labels=order, average="macro", zero_division=0)))
            samples["balanced_accuracy"].append(float(balanced_accuracy_score(truth, predicted)) if complete else None)
            auc = None
            if complete:
                if c["binary"]:
                    auc = roc_auc_score(truth == "irreversible", p[ix, list(classes).index("irreversible")])
                else:
                    sorted_classes = np.argsort(classes)
                    auc = roc_auc_score(truth, p[ix][:, sorted_classes], labels=classes[sorted_classes], multi_class="ovr", average="macro")
            samples["AUROC"].append(auc)
        for key, values in samples.items():
            check_interval(condition["selected"]["metrics"][key], values, len(names))
            independent_primary_checks += 1
    behavior = stats["RQ3"]["point_and_CI"]
    if behavior:
        pairs = summary["RQ3"]["pairs"]
        names = sorted({p["template_id"] for p in pairs})
        members = {g: [i for i, p in enumerate(pairs) if p["template_id"] == g] for g in names}
        draws = np.random.default_rng(9001).integers(0, len(names), size=(1000, len(names)))
        for key in ("K", "A", "gap", "sensitivity", "C_R", "C_I"):
            values = np.array([p[key] for p in pairs])
            resampled = [float(values[np.concatenate([members[names[k]] for k in draw])].mean()) for draw in draws]
            assert np.isclose(behavior[key]["mean"], values.mean())
            check_interval(behavior[key], resampled, len(names))
    receipt = {"status": "PASS_INDEPENDENT_STATISTICS_CHECK", "statistics_sha256": sha(args.statistics),
               "point_metric_checks": point_checks, "primary_intervals_independently_recomputed": independent_primary_checks,
               "explicit_rows_per_bootstrap_instead_of_weighted_confusion": True,
               "bootstrap_draws": 1000, "bootstrap_seed": 9001,
               "behavior_intervals_recomputed_from_joint_template_draws": bool(behavior),
               "human_agreement_created": False}
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
