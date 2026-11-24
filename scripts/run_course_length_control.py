"""Pre-test exploratory prompt-length baseline, with no further model calls."""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from src.data.course import check_access  # noqa: E402
from src.models.feature_store import sha  # noqa: E402
from src.probes.course import labels_for, partitions  # noqa: E402
from src.probes.linear import LABEL_ORDER, choose_linear, metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("select", "evaluate"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    dataset = ROOT / "data/course_fast_track/v2/sandbox_pool.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    report = json.loads((args.features / "report.json").read_text())
    assert report["status"] == "PASS_INFERENCE_COMPLETE"
    records = {}
    for shard in report["shards"]:
        for index, key in enumerate(shard["ids"]):
            with gzip.open(args.features / shard["path"] / f"prompt_{index:03d}.json.gz", "rt") as handle:
                record = json.load(handle)
            assert record["id"] == key
            records[key] = record
    assert set(records) == {r["id"] for r in rows}
    x = np.array([[records[r["id"]]["prompt_tokens_unpadded"]] for r in rows], dtype=np.float32)
    paired = {}
    for row in rows:
        if row.get("minimal_pair_id"):
            paired.setdefault(row["minimal_pair_id"], []).append(row)
    pair_audit = []
    for key, siblings in paired.items():
        assert len(siblings) == 2
        a, b = [records[r["id"]] for r in siblings]
        assert siblings[0]["action_text"] == siblings[1]["action_text"]
        assert a["selected_action_token_ids"] == b["selected_action_token_ids"]
        pair_audit.append({"pair_id": key, "ids": [r["id"] for r in siblings],
                           "labels": [r["label"] for r in siblings],
                           "prompt_token_difference": b["prompt_tokens_unpadded"] - a["prompt_tokens_unpadded"]})
    source = {"dataset_sha256": sha(dataset), "features_report_sha256": sha(args.features / "report.json"),
              "script_sha256": sha(Path(__file__)), "linear_source_sha256": sha(ROOT / "src/probes/linear.py"),
              "scope": "exploratory_pre_test_length_only_nuisance_control_not_registered_primary_contrast"}
    if args.mode == "select":
        if args.output.exists():
            raise ValueError("Choose a new length-control selection directory")
        args.output.mkdir(parents=True)
        args.models.mkdir(parents=True, exist_ok=True)
        selections = []
        for seed in (1729, 2718, 31415):
            manifest = json.loads((ROOT / f"data/course_fast_track/v2/splits/pool_1800/seed_{seed}.json").read_text())
            for binary in (True, False):
                condition = dict(horizon=3, binary=binary, subset="all")
                ix = partitions(rows, manifest, condition)
                for partition in ("train", "development"):
                    access = check_access(rows, manifest, [rows[i]["id"] for i in ix[partition]], partition)
                    access.update(stage="probe_fit", control="exploratory_prompt_length", timestamp_unix=time.time(),
                                  before_model_output=True, binary=binary)
                    with (args.output / "access.jsonl").open("a") as handle:
                        handle.write(json.dumps(access) + "\n")
                y = labels_for(rows, 3)
                order = ("reversible", "irreversible") if binary else LABEL_ORDER
                model, choice = choose_linear(x[ix["train"]], y[ix["train"]],
                    x[ix["development"]], y[ix["development"]], seed=seed, label_order=order)
                name = f"s{seed}_{'binary' if binary else 'threeway'}"
                target = args.models / (name + ".joblib")
                joblib.dump(model, target, compress=3)
                selections.append({"id": name, "seed": seed, "binary": binary, "choice": choice,
                    "model_sha256": sha(target), "label_order": list(order),
                    "training_ids": [rows[i]["id"] for i in ix["train"]],
                    "development_ids": [rows[i]["id"] for i in ix["development"]]})
        value = {**source, "status": "ALL_LENGTH_SELECTIONS_LOCKED_BEFORE_TEST", "completed_unix": time.time(),
                 "selections": selections, "pair_tokenization_audit": pair_audit,
                 "paired_actions_have_identical_selected_token_ids": True,
                 "equal_length_pair_count": sum(r["prompt_token_difference"] == 0 for r in pair_audit)}
        (args.output / "selection_lock.json").write_text(json.dumps(value, indent=2) + "\n")
        print("Locked six length-only fits and verified token identity for", len(pair_audit), "pairs")
    else:
        lock = json.loads((args.output / "selection_lock.json").read_text())
        assert all(lock[key] == value for key, value in source.items())
        target = args.output / "test_predictions.json"
        if target.exists():
            raise ValueError("Never overwrite length-control test outcomes")
        output = []
        for selected in lock["selections"]:
            model_path = args.models / (selected["id"] + ".joblib")
            assert sha(model_path) == selected["model_sha256"]
            model = joblib.load(model_path)
            manifest = json.loads((ROOT / f"data/course_fast_track/v2/splits/pool_1800/seed_{selected['seed']}.json").read_text())
            ix = partitions(rows, manifest, dict(horizon=3, binary=selected["binary"], subset="all"))["test"]
            access = check_access(rows, manifest, [rows[i]["id"] for i in ix], "test")
            access.update(stage="probe_evaluation", control="exploratory_prompt_length", timestamp_unix=time.time(),
                          before_model_output=True, selection_lock_sha256=sha(args.output / "selection_lock.json"))
            with (args.output / "access.jsonl").open("a") as handle:
                handle.write(json.dumps(access) + "\n")
            y = labels_for(rows, 3)[ix]
            predicted, proba = model.predict(x[ix]), model.predict_proba(x[ix])
            output.append({"id": selected["id"], "seed": selected["seed"], "binary": selected["binary"],
                "ids": [rows[i]["id"] for i in ix], "truth": y.tolist(), "predictions": predicted.tolist(),
                "probabilities": proba.tolist(), "classes": model.classes_.tolist(),
                "metrics": metrics(y, predicted, proba, model.classes_, selected["label_order"])})
        value = {**source, "status": "LENGTH_CONTROL_TEST_EVALUATED", "evaluated_unix": time.time(),
                 "selection_lock_sha256": sha(args.output / "selection_lock.json"), "records": output}
        target.write_text(json.dumps(value, indent=2) + "\n")
        print("Evaluated the six previously locked nuisance controls")


if __name__ == "__main__":
    main()
