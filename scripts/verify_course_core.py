"""Offline verification of actual core arrays, offsets, selections and predictions."""

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src.models.alignment import action_positions  # noqa: E402
from src.models.course_native import feature_messages  # noqa: E402
from src.models.feature_store import KEYS, sha  # noqa: E402
from src.probes.course import condition_id, conditions, labels_for, partitions, verb_labels  # noqa: E402


def verify_features(root, rows):
    report = json.loads((root / "report.json").read_text())
    assert report["status"] == "PASS_INFERENCE_COMPLETE"
    assert report["ids"] == [r["id"] for r in rows]
    assert report["model_calls_completed"] == len(rows)
    checkpoint = json.loads((ROOT / report["execution"]["checkpoint_receipt"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(ROOT / checkpoint["snapshot"], local_files_only=True,
                                              trust_remote_code=False)
    baseline = report["system_memory_baseline"]
    guards = [json.loads(line) for line in (root / "resource_guards.jsonl").read_text().splitlines()]
    assert all(r["swap_used_bytes"] - baseline["swap_used_bytes"] <= 512 * 1024**2 and
               r["memory_free_percentage"] >= 10 and r["free_disk_bytes"] >= 2 * 1024**3 for r in guards)
    events = [json.loads(line) for line in (root / "case_events.jsonl").read_text().splitlines()]
    assert [(e["id"], e["event"]) for e in events] == [(r["id"], kind) for r in rows for kind in ("started", "completed")]
    count, feature_bytes, prompt_bytes, times = 0, 0, 0, []
    for shard in report["shards"]:
        directory = root / shard["path"]
        assert sha(directory / "features.npz") == shard["sha256"]
        assert json.loads((directory / "complete.json").read_text())["sha256"] == shard["sha256"]
        n = len(shard["ids"])
        assert shard["ids"] == [r["id"] for r in rows[count:count + n]]
        with np.load(directory / "features.npz", allow_pickle=False) as arrays:
            for key in KEYS:
                expected = (25, n, 896) if key in ("last", "mean") else (n, 896)
                assert arrays[key].shape == expected and arrays[key].dtype == np.float32
                assert np.isfinite(arrays[key]).all()
        assert not list(directory.glob("row_*.npz")), "Unnecessary row work files retained"
        for offset, row in enumerate(rows[count:count + n]):
            path = directory / f"prompt_{offset:03d}.json.gz"
            with gzip.open(path, "rt") as handle:
                record = json.load(handle)
            assert record["id"] == row["id"] and record["messages"] == feature_messages(row)
            rendered = tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
            assert record["rendered_prompt"] == rendered and record["input_ids"] == encoded["input_ids"]
            assert record["offset_mapping"] == [list(pair) for pair in encoded["offset_mapping"]]
            start, end = record["action_span"]
            assert record["rendered_prompt"][start:end] == row["action_text"]
            positions = action_positions(record["offset_mapping"], record["attention_mask"], start, end)
            assert positions == record["action_positions"]
            assert [record["input_ids"][i] for i in positions] == record["selected_action_token_ids"]
            assert record["prompt_tokens_unpadded"] == sum(record["attention_mask"]) <= 2048
            times.append(record["seconds"])
            prompt_bytes += path.stat().st_size
        feature_bytes += (directory / "features.npz").stat().st_size
        count += n
    assert count == len(rows)
    accesses = [json.loads(line) for line in (root / "access.jsonl").read_text().splitlines()]
    first_start = {e["id"]: e["timestamp_unix"] for e in events if e["event"] == "started"}
    for seed in (1729, 2718, 31415):
        observed = []
        for access in accesses:
            if access["seed"] == seed:
                observed += access["ids"]
                assert all(access["timestamp_unix"] <= first_start[i] for i in access["ids"])
        assert sorted(observed) == sorted(first_start)
    return {"status": "PASS_FEATURE_INTEGRITY", "N": count, "shards": len(report["shards"]),
            "report_sha256": sha(root / "report.json"), "compressed_feature_bytes": feature_bytes,
            "compressed_prompt_bytes": prompt_bytes, "elapsed_seconds": report["elapsed_seconds"],
            "feature_seconds_min_median_max": [min(times), float(np.median(times)), max(times)],
            "maximum_swap_growth_bytes": max(r["swap_growth_bytes"] for r in guards),
            "minimum_free_memory_percentage": min(r["memory_free_percentage"] for r in guards),
            "process_peak_rss_bytes": report["memory"]["process_lifetime_peak_rss_bytes"]}


def independently_measure(record, label_order):
    truth = np.array(record["truth"])
    predicted = np.array(record["predictions"])
    proba = np.array(record["probabilities"])
    classes = np.array(record["classes"])
    assert len(record["ids"]) == len(set(record["ids"])) == len(truth)
    assert predicted.shape == truth.shape and proba.shape == (len(truth), len(classes))
    assert np.isfinite(proba).all() and np.allclose(proba.sum(1), 1) and (proba >= 0).all() and (proba <= 1).all()
    expected = {"N": len(truth), "accuracy": accuracy_score(truth, predicted),
                "balanced_accuracy": balanced_accuracy_score(truth, predicted),
                "macro_F1": f1_score(truth, predicted, labels=list(label_order), average="macro", zero_division=0)}
    if set(truth) == set(classes) == set(label_order):
        if len(label_order) == 2:
            expected["AUROC"] = roc_auc_score(truth == "irreversible", proba[:, list(classes).index("irreversible")])
        else:
            order = np.argsort(classes)
            expected["AUROC"] = roc_auc_score(truth, proba[:, order], labels=classes[order],
                                              multi_class="ovr", average="macro")
    else:
        assert record["metrics"]["AUROC"] is None
    assert all(np.isclose(record["metrics"][key], value, rtol=0, atol=1e-12) for key, value in expected.items())


def verify_probes(root, rows, models_root=None):
    lock = json.loads((root / "selection_lock.json").read_text())
    evaluation = json.loads((root / "test_evaluation.json").read_text())
    assert lock["status"] == "ALL_SELECTIONS_LOCKED_BEFORE_TEST_EVALUATION"
    assert evaluation["status"] == "PASS_TEST_EVALUATION_COMPLETE"
    assert evaluation["selection_lock_sha256"] == sha(root / "selection_lock.json")
    assert evaluation["predictions_sha256"] == sha(root / "test_predictions.jsonl.gz")
    assert len(lock["receipts"]) == 180
    manifests = {seed: json.loads((ROOT / f"data/course_fast_track/v2/splits/pool_1800/seed_{seed}.json").read_text())
                 for seed in (1729, 2718, 31415)}
    accesses = [json.loads(line) for line in (root / "access.jsonl").read_text().splitlines()]
    assert all(e["timestamp_unix"] >= lock["completed_at_unix"] for e in accesses if e["stage"] == "probe_evaluation")
    checked, unavailable = 0, []
    with gzip.open(root / "test_predictions.jsonl.gz", "rt") as handle:
        actual = [json.loads(line) for line in handle]
    assert [r["id"] for r in actual] == [condition_id(c) for c in conditions()]
    for condition, result, receipt in zip(conditions(), actual, lock["receipts"]):
        assert result["condition"] == condition and receipt["id"] == result["id"]
        path = root / (result["id"] + ".json")
        assert sha(path) == receipt["selection_sha256"]
        selection = json.loads(path.read_text())
        if models_root is not None:
            model_path = models_root / (result["id"] + ".joblib")
            assert sha(model_path) == receipt["model_sha256"]
            models = joblib.load(model_path)
            assert models["fingerprint"] == lock["fingerprint"]
            for layer, model in models["models"].items():
                assert list(model.named_steps) == ["standardscaler", "logisticregression"]
                logistic = model.named_steps["logisticregression"]
                assert logistic.penalty == "l2" and logistic.solver == "lbfgs"
                assert logistic.class_weight == "balanced" and logistic.max_iter == 5000
                assert logistic.C == selection["layers"][str(layer)]["selected_C"]
        assert selection["test_data_used_for_selection"] is False
        index = partitions(rows, manifests[condition["seed"]], condition)["test"]
        y = labels_for(rows, condition["horizon"])
        if condition["control"] == "verb_777":
            y, mapping = verb_labels(rows, selection["label_order"])
            assert mapping == selection["verb_mapping"]
        for layer, choice in selection["layers"].items():
            best = min(choice["tried"], key=lambda r: (-r["development"]["macro_F1"], r["C"]))
            assert choice["selected_C"] == best["C"]
        if selection["layers"]:
            selected = min(selection["layers"], key=lambda k:
                           (-selection["layers"][k]["development"]["balanced_accuracy"], int(k)))
            assert int(selected) == selection["selected_layer"] == result["selected_layer"]
        else:
            assert result["status"] == "NOT_ESTIMABLE_SINGLE_TRAINING_CLASS"
            unavailable.append(result["id"])
        records = list(result["layers"].values())
        if condition["control"] == "ordinary":
            records += [result["lexical"], result["majority"]]
        for record in records:
            assert record["ids"] == [rows[i]["id"] for i in index]
            assert record["truth"] == y[index].tolist()
            independently_measure(record, selection["label_order"])
            checked += 1
    return {"status": "PASS_PROBE_SELECTION_AND_METRICS", "conditions": 180,
            "independently_recomputed_metric_records": checked, "unavailable_controls": unavailable,
            "predictions_sha256": evaluation["predictions_sha256"],
            "all_seed_selection_before_any_test_performance": True}


def verify_length(root, rows, main_root):
    lock = json.loads((root / "selection_lock.json").read_text())
    evaluated = json.loads((root / "test_predictions.json").read_text())
    assert lock["status"] == "ALL_LENGTH_SELECTIONS_LOCKED_BEFORE_TEST"
    assert len(lock["selections"]) == len(evaluated["records"]) == 6
    assert evaluated["selection_lock_sha256"] == sha(root / "selection_lock.json")
    access = [json.loads(line) for line in (main_root / "access.jsonl").read_text().splitlines()]
    assert all(lock["completed_unix"] <= r["timestamp_unix"] for r in access if r["stage"] == "probe_evaluation")
    own_access = [json.loads(line) for line in (root / "access.jsonl").read_text().splitlines()]
    assert all(lock["completed_unix"] <= r["timestamp_unix"] for r in own_access if r["stage"] == "probe_evaluation")
    assert lock["paired_actions_have_identical_selected_token_ids"] is True
    assert len(lock["pair_tokenization_audit"]) == 810
    for selected, record in zip(lock["selections"], evaluated["records"]):
        assert selected["id"] == record["id"]
        manifest = json.loads((ROOT / f"data/course_fast_track/v2/splits/pool_1800/seed_{selected['seed']}.json").read_text())
        index = partitions(rows, manifest, dict(horizon=3, binary=selected["binary"], subset="all"))["test"]
        assert record["ids"] == [rows[i]["id"] for i in index]
        assert record["truth"] == labels_for(rows, 3)[index].tolist()
        independently_measure(record, selected["label_order"])
        best = min(selected["choice"]["tried"], key=lambda r: (-r["development"]["macro_F1"], r["C"]))
        assert selected["choice"]["selected_C"] == best["C"]
    return {"status": "PASS_EXPLORATORY_LENGTH_CONTROL", "N_conditions": 6,
            "selection_locked_before_main_and_auxiliary_test_evaluation": True,
            "predictions_sha256": sha(root / "test_predictions.json"),
            "registered_primary_contrast_changed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("features", "probes", "length"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--main-probes", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new verification receipt")
    rows = [json.loads(line) for line in (ROOT / "data/course_fast_track/v2/sandbox_pool.jsonl").read_text().splitlines()]
    if args.mode == "features":
        result = verify_features(args.root, rows)
    elif args.mode == "probes":
        result = verify_probes(args.root, rows, args.models)
    else:
        if args.main_probes is None:
            raise ValueError("Main-probe access log required for the pre-test timing check")
        result = verify_length(args.root, rows, args.main_probes)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
