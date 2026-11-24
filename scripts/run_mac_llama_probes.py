"""Fit and score the locked sandbox probes for the cached Llama feature condition."""

import argparse
import gzip
import json
import os
import time
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from course_llm_audit import ROOT, load_json, save, sha  # noqa: E402
from prepare_mac_llama_features import DATA, MODEL_DIGEST, rows  # noqa: E402

from src.data.course import check_access  # noqa: E402
from src.probes.course import condition_id, conditions, labels_for, partitions, verb_labels  # noqa: E402
from src.probes.linear import (  # noqa: E402
    LABEL_ORDER,
    choose_layer,
    choose_linear,
    metrics,
    training_majority,
)
from src.probes.row_space import fit_in_span, training_space  # noqa: E402
from src.probes.statistics import bootstrap_classification, bootstrap_mean  # noqa: E402

SEEDS = (1729, 2718, 31415)
RI = ("reversible", "irreversible")


def validate_features(root, preflight):
    inputs, outputs = root / "input360", root / "features360"
    assert load_json(outputs / "process.json")["status"] == "PASS"
    manifest = load_json(inputs / "manifest.json")
    assert manifest["model_blob_sha256"] == MODEL_DIGEST and manifest["preflight_sha256"] == sha(preflight)
    parent = DATA / "sandbox_pool.jsonl"
    by_id = {r["id"]: r for r in rows(parent)}
    assert manifest["data_sha256"] == sha(parent)
    prompts = manifest["cases"]
    data = [by_id[case["id"]] for case in prompts]
    token_log = rows(outputs / "stdout.log")
    assert len(prompts) == len(token_log) == 360
    pairs = {}
    for i, (case, record) in enumerate(zip(prompts, token_log)):
        assert case["index"] == record["index"] == i and record["layers"] == 29 and record["width"] == 3072
        path = inputs / f"{i}.prompt"
        begin, end = case["action_utf8_span"]
        assert sha(path) == case["prompt_sha256"] and path.read_bytes()[begin:end] == data[i]["action_text"].encode()
        assert record["action_positions"] and max(record["action_positions"]) < record["tokens"] - 1
        if case["pair"]:
            pairs.setdefault(case["pair"], []).append(record["action_tokens"])
    assert len(pairs) == 162 and all(len(v) == 2 and v[0] == v[1] for v in pairs.values())
    memory = rows(outputs / "memory.jsonl")
    raw = outputs / "pooled.f32"
    shape = (360, 2, 29, 3072)
    assert raw.stat().st_size == np.prod(shape) * 4
    vectors = np.memmap(raw, mode="r", dtype="<f4", shape=shape)
    assert np.isfinite(vectors).all()
    receipt = {"scope": manifest["study"], "shape": shape, "paired_action_ids_verified": len(pairs),
               "raw_sha256": sha(raw), "input_manifest_sha256": sha(inputs / "manifest.json"),
               "token_log_sha256": sha(outputs / "stdout.log"), "preflight_sha256": sha(preflight),
               "runtime_seconds_at_last_sample": memory[-1]["elapsed_seconds"],
               "lowest_memory_free_percentage": min(m["memory_free_percentage"] for m in memory),
               "maximum_added_swap_bytes": max(m["added_swap_bytes"] for m in memory)}
    return data, vectors, receipt


def score(model, values, ids, truth, order):
    predictions = model.predict(values)
    probabilities = model.predict_proba(values)
    return {"ids": ids, "truth": truth.tolist(), "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(), "classes": model.classes_.tolist(),
            "metrics": metrics(truth, predictions, probabilities, model.classes_, order)}


def select(data, vectors, splits, output, conditions_list):
    choices = {}
    texts = [r["action_text"] for r in data]
    grouped = {}
    for condition in conditions_list:
        index = partitions(data, splits[condition["seed"]], condition)
        train, dev = index["train"], index["development"]
        key = (condition["pooling"], tuple(train), tuple(dev))
        grouped.setdefault(key, []).append(condition)
    for (pooling, train_ids, dev_ids), group in grouped.items():
        train, dev = np.array(train_ids, dtype=int), np.array(dev_ids, dtype=int)
        fit_group(data, vectors, output, texts, pooling, train, dev, group, choices, len(conditions_list))
    return [choices[condition_id(condition)] for condition in conditions_list]


def fit_group(data, vectors, output, texts, pooling, train, dev, group, choices, total):
    pool = 0 if pooling == "last" else 1
    spaces = None
    for condition in group:
        order = RI if condition["binary"] else LABEL_ORDER
        y = labels_for(data, condition["horizon"])
        control = condition["control"]
        verb_map = None
        if control == "verb_777":
            y, verb_map = verb_labels(data, order)
        target = y[train]
        if control.startswith("shuffle_"):
            target = np.random.default_rng(int(control.split("_")[1])).permutation(target)
        name = condition_id(condition)
        path = output / f"{name}.json"
        result = {"id": name, "condition": condition, "order": order, "verb_mapping": verb_map,
                  "train_ids": [data[i]["id"] for i in train], "dev_ids": [data[i]["id"] for i in dev],
                  "evaluated_test": False}
        if path.exists():
            previous = load_json(path)
            assert all(previous[key] == value for key, value in json.loads(json.dumps(result)).items())
            if previous["status"] == "FIT":
                assert sha(output / f"{name}.joblib") == previous["fit_sha256"]
            choices[name] = {"id": name, "record_sha256": sha(path), "status": previous["status"]}
            continue
        if len(set(target)) < 2 or not len(dev):
            result["status"] = "NOT_ESTIMABLE_TRAIN_OR_DEV_SUPPORT"
        else:
            if spaces is None:
                spaces = [training_space(vectors[train, pool, layer], vectors[dev, pool, layer])
                          for layer in range(vectors.shape[2])]
            reports, models = {}, {}
            for layer, space in enumerate(spaces):
                models[layer], reports[layer] = fit_in_span(space, target, y[dev], condition["seed"], order)
            lexical = None
            if control == "ordinary":
                lexical, lexical_report = choose_linear([texts[i] for i in train], target,
                    [texts[i] for i in dev], y[dev], seed=condition["seed"], lexical=True, label_order=order)
                result.update(lexical=lexical_report, majority=training_majority(target, order))
            fitted = output / f"{name}.joblib"
            with fitted.open("xb") as handle:
                joblib.dump({"layers": models, "lexical": lexical}, handle)
            result.update(status="FIT", selected_layer=choose_layer(reports), layer_choice=reports,
                          fit_sha256=sha(fitted))
        save(path, result)
        choices[name] = {"id": name, "record_sha256": sha(path), "status": result["status"]}
        if len(choices) % 10 == 0:
            print(f"Selection {len(choices)}/{total}", flush=True)


def summaries(data, record, groups, condition, reference=None):
    lookup = {row["id"]: row for row in data}
    order = RI if condition["binary"] else LABEL_ORDER
    result = bootstrap_classification(record, groups, order, reference)
    by_pair = {}
    for key, truth, prediction, group in zip(record["ids"], record["truth"], record["predictions"], groups):
        pair = lookup[key]["minimal_pair_id"]
        if pair:
            by_pair.setdefault(pair, []).append((truth, prediction, group))
    paired = [members for members in by_pair.values() if len(members) == 2 and members[0][0] != members[1][0]]
    if paired:
        assert all(a[2] == b[2] for a, b in paired)
        values = [int(a[0] == a[1] and b[0] == b[1]) for a, b in paired]
        result["both_pair_endpoints_correct"] = {**bootstrap_mean(values, [p[0][2] for p in paired]),
                                                 "correct_pairs": sum(values)}
    return result


def evaluate(data, vectors, splits, output, conditions_list, choices):
    texts = [r["action_text"] for r in data]
    primary = []
    with gzip.open(output / "test_predictions.jsonl.gz", "wt") as raw:
        for condition, choice in zip(conditions_list, choices):
            name = condition_id(condition)
            assert choice["id"] == name
            selected = load_json(output / f"{name}.json")
            assert sha(output / f"{name}.json") == choice["record_sha256"]
            manifest = splits[condition["seed"]]
            test = partitions(data, manifest, condition)["test"]
            ids = [data[i]["id"] for i in test]
            order = RI if condition["binary"] else LABEL_ORDER
            y = labels_for(data, condition["horizon"])
            if selected["verb_mapping"] is not None:
                y, _ = verb_labels(data, order)
            record = {"id": name, "condition": condition, "status": selected["status"], "assigned_test_ids": ids}
            if choice["status"] == "FIT" and len(test):
                fit = output / f"{name}.joblib"
                assert sha(fit) == selected["fit_sha256"]
                trained = joblib.load(fit)
                pooling = 0 if condition["pooling"] == "last" else 1
                record["selected_layer"] = selected["selected_layer"]
                record["layers"] = {key: score(model, vectors[test, pooling, key], ids, y[test], order)
                                    for key, model in trained["layers"].items()}
                if condition["control"] == "ordinary":
                    lexical = score(trained["lexical"], [texts[i] for i in test], ids, y[test], order)
                    majority = selected["majority"]
                    p = np.zeros((len(test), len(order)))
                    p[:, order.index(majority)] = 1
                    prediction = [majority] * len(test)
                    record["lexical"] = lexical
                    record["majority"] = {"ids": ids, "truth": y[test].tolist(), "predictions": prediction,
                                          "probabilities": p.tolist(), "classes": order,
                                          "metrics": metrics(y[test], prediction, p, order, order)}
                    if condition["horizon"] == 3 and condition["pooling"] == "last":
                        groups = [manifest["components"][item] for item in ids]
                        chosen = record["layers"][record["selected_layer"]]
                        primary.append({"id": name, "N": len(ids), "condition": condition,
                                        "selected_layer": record["selected_layer"],
                                        "probe": summaries(data, chosen, groups, condition, lexical),
                                        "lexical": summaries(data, lexical, groups, condition),
                                        "majority": summaries(data, record["majority"], groups, condition)})
            raw.write(json.dumps(record, allow_nan=False) + "\n")
    return primary


def run(args):
    started = time.monotonic()
    data, vectors, receipt = validate_features(args.features, args.preflight)
    frozen = load_json(DATA / "freeze_manifest.json")["files"]
    paths = {seed: DATA / f"splits/subset_360/seed_{seed}.json" for seed in SEEDS}
    assert all(sha(path) == frozen[str(path.relative_to(DATA))]["sha256"] for path in paths.values())
    splits = {seed: load_json(path) for seed, path in paths.items()}
    if args.resume:
        assert load_json(args.output / "feature_verification.json") == json.loads(json.dumps(receipt))
        assert all(not (args.output / filename).exists() for filename in
                   ("selection_lock.json", "test_access.json", "test_predictions.jsonl.gz"))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "feature_verification.json", receipt)
    for split in splits.values():
        for partition in ("train", "development"):
            check_access(data, split, split["indices"][partition], partition)
    conditions_list = conditions()
    selected = select(data, vectors, splits, args.output, conditions_list)
    lock = {"scope": "retrospective cached Llama quantized sandbox", "choices": selected,
            "split_sha256": {seed: sha(path) for seed, path in paths.items()}, "no_test_scores_at_selection": True,
            "features_sha256": receipt["raw_sha256"], "analysis_source_sha256": sha(Path(__file__)),
            "row_space_source_sha256": sha(ROOT / "src/probes/row_space.py"),
            "protocol_sha256": sha(ROOT / "docs/MAC_LLAMA_RETROSPECTIVE_PROTOCOL.md")}
    lock_file = args.output / "selection_lock.json"
    save(lock_file, lock)
    accesses = [check_access(data, split, split["indices"]["test"], "test") for split in splits.values()]
    save(args.output / "test_access.json", {"selection_lock_sha256": sha(lock_file), "access": accesses})
    primary = evaluate(data, vectors, splits, args.output, conditions_list, selected)
    overlaps = {f"{a}-{b}": len(set(splits[a]["indices"]["test"]) & set(splits[b]["indices"]["test"]))
                for a, b in ((1729, 2718), (1729, 31415), (2718, 31415))}
    save(args.output / "summary.json", {"status": "PASS_REDUCED_LLAMA_PROBE_EXECUTION", "conditions": len(selected),
         "fit_conditions": sum(choice["status"] == "FIT" for choice in selected), "primary": primary,
         "test_overlap_counts": overlaps, "test_predictions_sha256": sha(args.output / "test_predictions.jsonl.gz"),
         "elapsed_seconds": time.monotonic() - started, "original_final_study_complete": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
