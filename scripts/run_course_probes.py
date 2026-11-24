"""Lock every training/development choice, then evaluate immutable saved probes."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from src.data.course import check_access  # noqa: E402
from src.models.feature_store import KEYS, sha  # noqa: E402
from src.probes.course import (  # noqa: E402
    condition_id,
    conditions,
    labels_for,
    partitions,
    verb_labels,
)
from src.probes.linear import (  # noqa: E402
    LABEL_ORDER,
    choose_layer,
    choose_linear,
    metrics,
    training_majority,
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_features(root, rows):
    report = json.loads((root / "report.json").read_text())
    assert report["status"] == "PASS_INFERENCE_COMPLETE"
    assert report["ids"] == [r["id"] for r in rows]
    observed = {key: [] for key in KEYS}
    ids = []
    for shard in report["shards"]:
        path = root / shard["path"] / "features.npz"
        assert sha(path) == shard["sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            for key in KEYS:
                assert np.isfinite(arrays[key]).all() and arrays[key].dtype == np.float32
                observed[key].append(arrays[key])
        ids.extend(shard["ids"])
    assert ids == report["ids"]
    return {key: np.concatenate(values, axis=1 if key in ("last", "mean") else 0)
            for key, values in observed.items()}


def prediction(pipeline, values, ids, truth, label_order):
    predictions = pipeline.predict(values)
    probabilities = pipeline.predict_proba(values)
    return {"ids": ids, "truth": truth.tolist(), "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(), "classes": pipeline.classes_.tolist(),
            "metrics": metrics(truth, predictions, probabilities, pipeline.classes_, label_order)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("select", "evaluate"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    execution = json.loads((ROOT / "configs/course_execution_native_mac_cpu_v1.json").read_text())
    data = ROOT / execution["selected_dataset"]
    assert sha(data) == execution["selected_dataset_sha256"]
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    manifests = {int(seed): json.loads((ROOT / value["path"]).read_text())
                 for seed, value in execution["split_manifests"].items()}
    paths = [Path(__file__), ROOT / "src/probes/course.py", ROOT / "src/probes/linear.py",
             ROOT / "docs/COURSE_CORE_PROTOCOL.md", ROOT / "configs/course_execution_native_mac_cpu_v1.json"]
    source = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    frozen = {"dataset_sha256": sha(data), "features_report_sha256": sha(args.features / "report.json"),
              "source_sha256": source, "conditions": conditions(),
              "seeds": [1729, 2718, 31415], "test_evaluated_at_selection": False,
              "python": sys.version, "platform": platform.platform(),
              "packages": {p: importlib.metadata.version(p) for p in
                           ("numpy", "scipy", "scikit-learn", "joblib")},
              "BLAS_thread_environment": {name: os.environ[name] for name in
                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}}
    fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    source_path = args.results / "source_lock.json"
    if source_path.exists():
        assert json.loads(source_path.read_text()) == frozen, "Resume changes source, features or conditions"
    else:
        assert args.mode == "select"
        save(source_path, frozen)
    features = load_features(args.features, rows)
    label_orders = {True: ("reversible", "irreversible"), False: LABEL_ORDER}
    if args.mode == "select":
        if (args.results / "test_evaluation.json").exists():
            raise ValueError("Cannot select after holdout evaluation")
        for manifest in manifests.values():
            for partition in ("train", "development"):
                record = check_access(rows, manifest, manifest["indices"][partition], partition)
                record.update(stage="probe_fit", before_any_probe_selection_output=True,
                              timestamp_unix=time.time())
                with (args.results / "access.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
        receipts = []
        for count, condition in enumerate(conditions(), 1):
            name = condition_id(condition)
            target = args.models / (name + ".joblib")
            selection_path = args.results / (name + ".json")
            if selection_path.exists():
                old = json.loads(selection_path.read_text())
                assert old["fingerprint"] == fingerprint and sha(target) == old["model_sha256"]
                receipts.append({"id": name, "selection_sha256": sha(selection_path), "model_sha256": sha(target)})
                continue
            index = partitions(rows, manifests[condition["seed"]], condition)
            train, dev = index["train"], index["development"]
            label_order = label_orders[condition["binary"]]
            y = labels_for(rows, condition["horizon"])
            control = condition["control"]
            mapping = None
            if control == "verb_777":
                y, mapping = verb_labels(rows, label_order)
            y_train = y[train].copy()
            if control.startswith("shuffle_"):
                y_train = np.random.default_rng(int(control.split("_")[1])).permutation(y_train)
            models, choices = {}, {}
            selection = {"condition": condition, "fingerprint": fingerprint, "verb_mapping": mapping,
                         "training_ids": [rows[i]["id"] for i in train],
                         "development_ids": [rows[i]["id"] for i in dev],
                         "training_target_counts": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
                         "label_order": list(label_order), "test_data_used_for_selection": False}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if len(set(y_train)) >= 2:
                    x = features[condition["pooling"]]
                    for layer in range(x.shape[0]):
                        models[layer], choices[layer] = choose_linear(x[layer, train], y_train,
                            x[layer, dev], y[dev], seed=condition["seed"], label_order=label_order)
                    selected = choose_layer(choices)
                    selection.update(status="SELECTED", selected_layer=selected, layers=choices)
                    if condition["subset"] == "all" and control == "ordinary":
                        selection["steering_block_candidate"] = choose_layer({k: v for k, v in choices.items() if k > 0})
                else:
                    selection.update(status="NOT_ESTIMABLE_SINGLE_TRAINING_CLASS", selected_layer=None, layers={})
                lexical = None
                if control == "ordinary":
                    text = [r["action_text"] for r in rows]
                    lexical, lexical_choice = choose_linear([text[i] for i in train], y_train,
                        [text[i] for i in dev], y[dev], seed=condition["seed"], lexical=True, label_order=label_order)
                    selection.update(lexical=lexical_choice, majority=training_majority(y_train, label_order))
            selection["warnings"] = sorted({str(w.message) for w in caught})
            temporary = target.with_suffix(".part")
            joblib.dump({"models": models, "lexical": lexical, "fingerprint": fingerprint}, temporary, compress=3)
            temporary.rename(target)
            selection.update(model_sha256=sha(target), elapsed_total_seconds=time.monotonic() - started)
            save(selection_path, selection)
            receipts.append({"id": name, "selection_sha256": sha(selection_path), "model_sha256": sha(target)})
            print(f"selection {count}/180: {name}; elapsed {time.monotonic() - started:.1f}s", flush=True)
        lock = {**frozen, "status": "ALL_SELECTIONS_LOCKED_BEFORE_TEST_EVALUATION", "fingerprint": fingerprint,
                "completed_at_unix": time.time(), "elapsed_seconds": time.monotonic() - started,
                "receipts": receipts, "N_conditions": len(receipts)}
        assert len(receipts) == 180
        save(args.results / "selection_lock.json", lock)
    else:
        lock_path = args.results / "selection_lock.json"
        locked = json.loads(lock_path.read_text())
        assert locked["status"] == "ALL_SELECTIONS_LOCKED_BEFORE_TEST_EVALUATION"
        assert locked["fingerprint"] == fingerprint and len(locked["receipts"]) == len(conditions())
        target = args.results / "test_predictions.jsonl.gz"
        if target.exists():
            raise ValueError("Never overwrite observed holdout results")
        for receipt in locked["receipts"]:
            assert sha(args.results / (receipt["id"] + ".json")) == receipt["selection_sha256"]
            assert sha(args.models / (receipt["id"] + ".joblib")) == receipt["model_sha256"]
        for manifest in manifests.values():
            record = check_access(rows, manifest, manifest["indices"]["test"], "test")
            record.update(stage="probe_evaluation", selection_lock_sha256=sha(lock_path),
                          timestamp_unix=time.time(), before_any_test_performance=True)
            with (args.results / "access.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        evaluations = []
        with gzip.open(target, "wt") as handle:
            for condition in conditions():
                name = condition_id(condition)
                selection = json.loads((args.results / (name + ".json")).read_text())
                saved = joblib.load(args.models / (name + ".joblib"))
                assert saved["fingerprint"] == fingerprint
                index = partitions(rows, manifests[condition["seed"]], condition)["test"]
                ids = [rows[i]["id"] for i in index]
                y = labels_for(rows, condition["horizon"])
                if condition["control"] == "verb_777":
                    y, mapping = verb_labels(rows, label_orders[condition["binary"]])
                    assert mapping == selection["verb_mapping"]
                label_order = label_orders[condition["binary"]]
                result = {"id": name, "condition": condition, "selected_layer": selection["selected_layer"],
                          "status": selection["status"], "layers": {}}
                for layer, model in saved["models"].items():
                    values = features[condition["pooling"]][layer, index]
                    result["layers"][str(layer)] = prediction(model, values, ids, y[index], label_order)
                if condition["control"] == "ordinary":
                    result["lexical"] = prediction(saved["lexical"], [rows[i]["action_text"] for i in index],
                                                   ids, y[index], label_order)
                    majority = selection["majority"]
                    proba = np.zeros((len(index), len(label_order)))
                    proba[:, list(label_order).index(majority)] = 1
                    result["majority"] = {"ids": ids, "truth": y[index].tolist(),
                        "predictions": [majority] * len(index), "probabilities": proba.tolist(),
                        "classes": list(label_order), "metrics": metrics(y[index], [majority] * len(index),
                                                                         proba, label_order, label_order)}
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                evaluations.append({"id": name, "status": result["status"], "N": len(ids)})
        save(args.results / "test_evaluation.json", {"status": "PASS_TEST_EVALUATION_COMPLETE",
            "selection_lock_sha256": sha(lock_path), "predictions_sha256": sha(target),
            "elapsed_seconds": time.monotonic() - started, "conditions": evaluations})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
