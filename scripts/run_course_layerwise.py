"""Measure supplemental uncertainty without reopening any model evaluation."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.probes.course import condition_id, conditions, partitions, verb_labels  # noqa: E402
from src.probes.layerwise import component_metrics  # noqa: E402
from src.probes.linear import LABEL_ORDER  # noqa: E402
from src.probes.statistics import group_weights, interval  # noqa: E402

BASE = Path("results/course_fast_track/native_mac_cpu_v1")
PREDICTIONS = BASE / "probes/test_predictions.jsonl.gz"
SELECTION = BASE / "probes/selection_lock.json"
EVALUATION = BASE / "probes/test_evaluation.json"
EXECUTION = Path("configs/course_execution_native_mac_cpu_v1.json")
IMPROVEMENT = Path("configs/course_improvement_v1.json")


def digest(path):
    with (ROOT / path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def binary_order(condition):
    return ("reversible", "irreversible") if condition["binary"] else LABEL_ORDER


def same(first, second):
    if first is None or second is None:
        assert first is second, (first, second)
    else:
        assert np.allclose(first, second, atol=1e-12, rtol=0), (first, second)


def bootstrap_design(groups):
    weights, design = group_weights(groups)
    _, first, membership = np.unique(groups, return_index=True, return_inverse=True)
    counts, inverse = np.unique(weights[:, first], axis=0, return_inverse=True)
    return np.vstack((np.ones(len(first), dtype=int), counts)), membership, inverse + 1, design


def summarize(values, inverse, design, estimate):
    return {"estimate": estimate, **interval(values[inverse], design["clusters"])}


def calculate():
    started = time.monotonic()
    plan = read(IMPROVEMENT)
    for mapping in ("protected_original_sha256", "protected_baseline_sha256"):
        assert all(digest(Path(path)) == expected for path, expected in plan[mapping].items()), mapping
    old = read(BASE / "statistics.json")
    assert old["source_sha256"][str(PREDICTIONS)] == digest(PREDICTIONS)
    evaluation = read(EVALUATION)
    assert evaluation["predictions_sha256"] == digest(PREDICTIONS)
    assert evaluation["selection_lock_sha256"] == digest(SELECTION)
    selection = read(SELECTION)
    assert selection["status"] == "ALL_SELECTIONS_LOCKED_BEFORE_TEST_EVALUATION"
    assert selection["conditions"] == conditions()
    execution = read(EXECUTION)
    dataset = Path(execution["selected_dataset"])
    assert digest(dataset) == execution["selected_dataset_sha256"] == selection["dataset_sha256"]
    rows = [json.loads(line) for line in (ROOT / dataset).read_text().splitlines()]
    manifests = {}
    for seed, item in execution["split_manifests"].items():
        assert digest(Path(item["path"])) == item["sha256"]
        manifests[int(seed)] = read(item["path"])
    with gzip.open(ROOT / PREDICTIONS, "rt") as handle:
        raw = list(map(json.loads, handle))
    assert [row["id"] for row in raw] == [condition_id(c) for c in conditions()]
    previous = {row["id"]: row for row in old["conditions"]}
    controls = {row["id"]: row for row in raw if row["condition"]["control"] == "verb_777"}
    labels_by_verb = {binary: verb_labels(rows, binary_order({"binary": binary}))[0]
                      for binary in (True, False)}
    result = []
    for source in raw:
        condition, name = source["condition"], source["id"]
        assert condition_id(condition) == name
        manifest = manifests[condition["seed"]]
        eligible = partitions(rows, manifest, condition)["test"]
        ids = [rows[index]["id"] for index in eligible]
        assert ids and len(set(ids)) == len(ids)
        groups = [manifest["components"][key] for key in ids]
        labels = binary_order(condition)
        if condition["control"] == "verb_777":
            truth = labels_by_verb[condition["binary"]][eligible].tolist()
        else:
            truth = [rows[index]["labels_by_horizon"][str(condition["horizon"])]["label"] for index in eligible]
        counts, membership, inverse, design = bootstrap_design(groups)
        before = previous[name]
        assert source["status"] == before["status"] == "SELECTED"
        assert source["selected_layer"] == before["selected_layer"]
        assert set(source["layers"]) == set(before["layer_accuracy"]) == {str(i) for i in range(25)}
        control = controls.get(name.removesuffix("_ordinary") + "_verb_777")
        selectivity_status = ("SAME_LAYER_PAIRED" if control else
                              "NO_MATCHING_SAVED_CONTROL" if condition["control"] == "ordinary" else
                              "NOT_APPLICABLE_TO_CONTROL")
        output = {"id": name, "condition": condition, "N": len(ids), "design": design,
                  "class_counts": dict(Counter(truth)), "locked_selected_layer": source["selected_layer"],
                  "selectivity_status": selectivity_status, "layers": {}}
        for layer, record in source["layers"].items():
            assert record["ids"] == ids and record["truth"] == truth
            values, complete = component_metrics(record, membership, counts, labels)
            measured = {}
            for key, samples in values.items():
                expected = record["metrics"][key]
                point = float(samples[0]) if np.isfinite(samples[0]) else None
                if key == "balanced_accuracy" and not complete[0]:
                    # The baseline reports an observed-class BA point; the full-label CI is undefined.
                    point = expected
                same(point, expected)
                measured[key] = summarize(samples, inverse, design, point)
            baseline = before["layer_accuracy"][layer]
            same(measured["accuracy"]["CI95"], baseline["CI95"])
            if int(layer) == source["selected_layer"]:
                for key, item in measured.items():
                    prior = before["selected"]["metrics"][key]
                    same(item["CI95"], prior["CI95"])
                    assert item["defined_draws"] == prior["defined_draws"]
            item = {"missing_true_class_draws": int((~complete[inverse]).sum()), "metrics": measured}
            if control:
                verb = control["layers"][layer]
                assert verb["ids"] == ids
                ordinary_correct = np.array(record["predictions"]) == truth
                verb_correct = np.array(verb["predictions"]) == verb["truth"]
                totals = np.bincount(membership)
                difference = ordinary_correct.astype(int) - verb_correct
                cluster_diff = np.bincount(membership, weights=difference)
                paired = (counts @ cluster_diff) / (counts @ totals)
                item["selectivity"] = summarize(paired, inverse, design, float(paired[0]))
            output["layers"][layer] = item
        result.append(output)
    paths = [Path(__file__).relative_to(ROOT), Path("src/probes/layerwise.py"), Path("src/probes/statistics.py"),
             Path("src/probes/course.py"), PREDICTIONS, SELECTION, EVALUATION, BASE / "statistics.json",
             EXECUTION, IMPROVEMENT, dataset, *(Path(item["path"]) for item in execution["split_manifests"].values())]
    return {"status": "COMPUTED_PENDING_INDEPENDENT_VERIFICATION", "scope": "post_result_reduced_course_supplement",
            "new_model_calls": 0, "probe_refits": 0, "human_labels_created": 0,
            "bootstrap_seed": 9001, "bootstrap_draws": 1000, "nominal_CI": "pointwise_percentile_95",
            "source_sha256": {str(path): digest(path) for path in paths},
            "environment": {"Python": platform.python_version(), "platform": platform.platform(),
                            **{name: importlib.metadata.version(name) for name in
                               ("numpy", "scipy", "scikit-learn")}},
            "elapsed_seconds": time.monotonic() - started, "conditions": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output; previous results are immutable")
    payload = json.dumps(calculate(), separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with args.output.open("xb") as handle:
        handle.write(gzip.compress(payload, mtime=0))
    print(f"Wrote {args.output} ({len(payload):,} bytes before compression).")


if __name__ == "__main__":
    main()
