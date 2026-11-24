"""Recount raw probe and stated outcomes on the locked retrospective IDs."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from course_llm_audit import load_json, save, sha
from prepare_mac_llama_features import DATA, rows
from run_mac_llama_probes import SEEDS, validate_features
from sklearn.metrics import cohen_kappa_score

from src.models.course_native import parse_stated, stated_messages
from src.probes.statistics import bootstrap_kappa, bootstrap_mean


def verify(root, preflight):
    data, _, feature_receipt = validate_features(root, preflight)
    probes, stated = root / "probes_row_space", root / "stated"
    assert feature_receipt == load_json(probes / "feature_verification.json") | {"shape": feature_receipt["shape"]}
    lookup = {row["id"]: row for row in data}
    summary = load_json(probes / "summary.json")
    lock = probes / "selection_lock.json"
    test_access = load_json(probes / "test_access.json")
    assert test_access["selection_lock_sha256"] == sha(lock)
    assert load_json(stated / "input_lock.json")["test_access_sha256"] == sha(probes / "test_access.json")
    predictions = probes / "test_predictions.jsonl.gz"
    assert sha(predictions) == summary["test_predictions_sha256"]
    with gzip.open(predictions, "rt") as stream:
        outcomes = {row["id"]: row for row in map(json.loads, stream)}
    choices = load_json(lock)["choices"]
    assert len(choices) == len(outcomes) == 180
    counts = Counter()
    for choice in choices:
        key = choice["id"]
        selected = probes / f"{key}.json"
        assert sha(selected) == choice["record_sha256"]
        selection = load_json(selected)
        assert sha(probes / f"{key}.joblib") == selection["fit_sha256"]
        record = outcomes[key]
        assert record["selected_layer"] == selection["selected_layer"] and len(record["layers"]) == 29
        ids = record["assigned_test_ids"]
        assert not set(ids) & set(selection["train_ids"] + selection["dev_ids"])
        condition = record["condition"]
        scope = next(part for part in test_access["access"] if part["seed"] == condition["seed"])
        assert set(ids) <= set(scope["ids"])
        expected = [lookup[item]["labels_by_horizon"][str(condition["horizon"])]["label"] for item in ids]
        for layer in record["layers"].values():
            assert layer["ids"] == ids
            if condition["control"] != "verb_777":
                assert layer["truth"] == expected
            assert abs(sum(a == b for a, b in zip(layer["truth"], layer["predictions"])) / len(ids)
                       - layer["metrics"]["accuracy"]) < 1e-12
            counts["layer_predictions"] += 1
    answers = load_json(stated / "result.json")
    assigned = load_json(stated / "input_lock.json")["assigned_ids"]
    receipts = rows(stated / "labels.jsonl")
    assert answers == {"status": "COMPLETE", "assigned": 96, "attempts": 96, "resident_after": []}
    assert [r["id"] for r in receipts] == assigned == sorted({i for p in test_access["access"] for i in p["ids"]})
    model_labels = {}
    for entry in receipts:
        key, number = entry["id"], entry["request"]
        request, response = (stated / f"{number:03d}.{kind}.json" for kind in ("request", "response"))
        assert load_json(request)["messages"] == stated_messages(lookup[key])
        reply = load_json(response)
        assert reply["done"] and sha(response) == entry["response_sha256"]
        label = parse_stated(reply["message"]["content"]) if reply["done_reason"] == "stop" else None
        assert label == entry["label"]
        model_labels[key] = label or "invalid"
    stated_summary = []
    for seed in SEEDS:
        split = load_json(DATA / f"splits/subset_360/seed_{seed}.json")
        record = outcomes[f"s{seed}_h3_threeway_last_all_ordinary"]
        chosen = record["layers"][str(record["selected_layer"])]
        ids = chosen["ids"]
        statuses = [model_labels[item] for item in ids]
        groups = [split["components"][item] for item in ids]
        correct = [int(a == b) for a,b in zip(statuses, chosen["truth"])]
        confusion = Counter((a, b) for a,b in zip(chosen["truth"], statuses))
        stated_summary.append({"seed": seed, "N": len(ids), "correct": sum(correct), "labels": dict(Counter(statuses)),
            "oracle_accuracy": bootstrap_mean(correct, groups),
            "agreement_with_three_way_probe": {"kappa": float(cohen_kappa_score(statuses, chosen["predictions"])),
                **bootstrap_kappa(statuses, chosen["predictions"], groups)},
            "confusion": [{"oracle": a, "response": b, "count": count} for (a, b), count in sorted(confusion.items())]})
    return {"scope": "retrospective sandbox, cached Llama Q4_K_M; original gates independent", "feature_verification":
            load_json(probes / "feature_verification.json"), "probe_condition_count": len(outcomes), "checks": dict(counts),
            "primary_probe": summary["primary"], "test_overlap_counts": summary["test_overlap_counts"],
            "stated": stated_summary, "stated_unique": {"N": len(assigned), "labels": dict(Counter(model_labels.values()))},
            "stated_resource_max_added_swap_bytes": max(r["added_swap_bytes"] for r in rows(stated / "memory.jsonl")),
            "test_predictions_sha256": sha(predictions), "selection_lock_sha256": sha(lock)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save(args.output, verify(args.root, args.preflight))
