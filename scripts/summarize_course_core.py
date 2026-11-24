"""Descriptive core joins only; cluster intervals and figures belong to Phase V."""

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from src.agent.course_metrics import audit_course_episode  # noqa: E402
from src.agent.course_summary import behavior_summary, kappa_record  # noqa: E402
from src.models.course_native import feature_messages, parse_stated  # noqa: E402
from src.models.feature_store import sha  # noqa: E402


def read_stated(root, expected_ids):
    rows = [json.loads(line) for line in (root / "stated.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == expected_ids
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(parse_stated(r["text"]) == r["parsed_label"] for r in rows)
    return {r["id"]: r for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--stated", type=Path, required=True)
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--length", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new summary path")
    dataset = ROOT / "data/course_fast_track/v2/sandbox_pool.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    stated_report = json.loads((args.stated / "report.json").read_text())
    assert stated_report["status"] == "PASS_INFERENCE_COMPLETE"
    stated = read_stated(args.stated, [r["id"] for r in rows])
    with gzip.open(args.probes / "test_predictions.jsonl.gz", "rt") as handle:
        conditions = [json.loads(line) for line in handle]
    by_id = {r["id"]: r for r in conditions}
    summaries, agreements = [], []
    for result in conditions:
        c = result["condition"]
        selected = str(result["selected_layer"])
        summary = {"id": result["id"], "condition": c, "status": result["status"],
                   "selected_layer": result["selected_layer"],
                   "selected_metrics": result["layers"][selected]["metrics"] if result["layers"] else None,
                   "layer_0_metrics": result["layers"].get("0", {}).get("metrics")}
        if c["control"] == "ordinary":
            summary.update(lexical=result["lexical"]["metrics"], majority=result["majority"]["metrics"])
            if c["horizon"] == 3 and c["subset"] == "all":
                prefix = result["id"].rsplit("_ordinary", 1)[0]
                shuffled = [by_id[prefix + f"_shuffle_{s}"] for s in range(101, 111)]
                values = [r["layers"][str(r["selected_layer"])]["metrics"] for r in shuffled if r["layers"]]
                summary["shuffled_controls"] = {"N_controls": len(values),
                    "mean": {key: float(np.mean([r[key] for r in values])) for key in ("accuracy", "balanced_accuracy", "macro_F1")},
                    "sample_SD": {key: float(np.std([r[key] for r in values], ddof=1)) for key in
                                  ("accuracy", "balanced_accuracy", "macro_F1")}}
                verb = by_id[prefix + "_verb_777"]
                if verb["layers"]:
                    control_accuracy = verb["layers"][str(verb["selected_layer"])]["metrics"]["accuracy"]
                    summary["selectivity_accuracy_difference"] = summary["selected_metrics"]["accuracy"] - control_accuracy
                else:
                    summary["selectivity_accuracy_difference"] = None
                    summary["selectivity_undefined_reason"] = verb["status"]
                if not c["binary"]:
                    record = result["layers"][selected]
                    joined = [{"id": key, "truth": truth, "probe": probe,
                               "stated": stated[key]["parsed_label"] or "invalid",
                               "request_id": stated[key]["request_id"]}
                              for key, truth, probe in zip(record["ids"], record["truth"], record["predictions"])]
                    valid = [r for r in joined if r["stated"] != "invalid"]
                    agreements.append({"seed": c["seed"], "pooling": c["pooling"], "N": len(joined),
                        "stated_all_ID_accuracy": sum(r["stated"] == r["truth"] for r in joined) / len(joined),
                        "valid_label_count": len(valid), "invalid_count": len(joined) - len(valid),
                        "valid_only_accuracy": sum(r["stated"] == r["truth"] for r in valid) / len(valid) if valid else None,
                        "probe_stated_kappa_all_ID": kappa_record([r["probe"] for r in joined], [r["stated"] for r in joined]),
                        "probe_stated_kappa_valid_only": kappa_record([r["probe"] for r in valid], [r["stated"] for r in valid]) if valid else None,
                        "confusion_truth_to_stated": dict(Counter(r["truth"] + ":" + r["stated"] for r in joined)),
                        "joined": joined})
        summaries.append(summary)
    data = ROOT / "data/course_fast_track/v2"
    tasks = [json.loads(line) for line in (data / "development_tasks.jsonl").read_text().splitlines()]
    behavior_report = json.loads((args.behavior / "report.json").read_text())
    audits, statuses = {}, Counter()
    for entry in behavior_report["episodes"]:
        task = next(t for t in tasks if t["id"] == entry["id"])
        episode = json.loads((args.behavior / entry["episode_file"]).read_text())
        audit = audit_course_episode(task, episode)
        assert audit == entry["audit"]
        audits[task["id"]] = audit
        statuses[episode["status"]] += 1
    development_stated = read_stated(args.behavior, [t["id"] for t in tasks])
    behavior = behavior_summary(tasks, {key: r["parsed_label"] for key, r in development_stated.items()}, audits)
    behavior["terminal_status_counts"] = dict(statuses)
    behavior["source_report_sha256"] = sha(args.behavior / "report.json")
    result = {"status": "DESCRIPTIVE_RESULTS_PENDING_PHASE_V_CLUSTER_STATISTICS", "human_judgments_created": 0,
              "model": "Qwen/Qwen2.5-0.5B-Instruct", "device": "Mac_CPU", "precision": "float32",
              "dataset_sha256": sha(dataset), "probe_conditions": summaries,
              "RQ2": agreements, "RQ3": behavior, "physical_transfer": "NOT_RUN_NO_CERTIFIED_PHYSICAL_DATA",
              "two_family_BF16_comparison": "NOT_RUN", "human_annotation": "NOT_PERFORMED",
              "source_sha256": {str(p): sha(p) for p in (args.probes / "test_predictions.jsonl.gz",
                  args.probes / "selection_lock.json", args.stated / "report.json", args.behavior / "report.json",
                  args.length / "selection_lock.json", args.length / "test_predictions.json")}}
    length = json.loads((args.length / "test_predictions.json").read_text())
    length_lock = json.loads((args.length / "selection_lock.json").read_text())
    assert length["status"] == "LENGTH_CONTROL_TEST_EVALUATED"
    assert length["selection_lock_sha256"] == sha(args.length / "selection_lock.json")
    assert len(length["records"]) == 6
    result["exploratory_length_control"] = length["records"]
    result["minimal_pair_token_audit"] = {"N_pairs": len(length_lock["pair_tokenization_audit"]),
        "equal_length_pairs": length_lock["equal_length_pair_count"], "candidate_token_identity": True}
    # Reuse only the independently verified Mac CPU development features, never
    # the separate DGX/MPS profiles. This joins all three measurements without
    # treating already exposed development states as additional holdout data.
    pilot = ROOT / "artifacts/course_fast_track/native_mac_cpu_pilot_v1/native_pilot"
    pilot_report = json.loads((pilot / "report.json").read_text())
    assert sha(pilot / "features.npz") == pilot_report["features"]["file_sha256"]
    assert pilot_report["settings"]["device"] == "cpu" and pilot_report["settings"]["CPU_threads"] == 4
    assert pilot_report["settings"]["model"] == behavior_report["execution"]["checkpoint"]
    assert pilot_report["settings"]["revision"] == behavior_report["execution"]["checkpoint_revision"]
    assert pilot_report["settings"]["precision"] == "float32" and pilot_report["settings"]["attention_backend"] == "eager"
    for package in ("torch", "transformers", "tokenizers", "numpy", "safetensors"):
        assert pilot_report["packages"][package] == behavior_report["packages"][package]
    assert pilot_report["source_sha256"]["src/models/course_native.py"] == behavior_report["execution"]["model_runner_source_sha256"]
    assert pilot_report["features"]["ids"] == [t["id"] for t in tasks]
    prompts = [json.loads(line) for line in (pilot / "feature_prompts.jsonl").read_text().splitlines()]
    assert [r["id"] for r in prompts] == [t["id"] for t in tasks]
    assert all(p["messages"] == feature_messages(t) for p, t in zip(prompts, tasks))
    name = "s1729_h3_threeway_last_all_ordinary"
    selected = json.loads((args.probes / (name + ".json")).read_text())
    model_path = args.models / (name + ".joblib")
    assert sha(model_path) == selected["model_sha256"]
    probe = joblib.load(model_path)["models"][selected["selected_layer"]]
    with np.load(pilot / "features.npz", allow_pickle=False) as saved:
        x = saved["last"][selected["selected_layer"]]
    predicted, probabilities = probe.predict(x), probe.predict_proba(x)
    result["development_representation_knowledge_behavior"] = {
        "scope": "previously_exposed_development_cases_not_heldout_accuracy_or_independent_evidence",
        "fixed_probe_condition": name, "selected_layer": selected["selected_layer"],
        "feature_source_sha256": sha(pilot / "features.npz"), "model_sha256": sha(model_path),
        "classes": probe.classes_.tolist(), "N": len(tasks),
        "rows": [{"id": t["id"], "truth": t["label"], "probe_label": str(predicted[i]),
                  "probe_probabilities": probabilities[i].tolist(),
                  "stated_label": development_stated[t["id"]]["parsed_label"],
                  "caution": audits.get(t["id"], {}).get("caution"),
                  "goal_success": audits.get(t["id"], {}).get("goal_success"),
                  "shortcut_executed": audits.get(t["id"], {}).get("shortcut_executed"),
                  "executed_irreversible_action": audits.get(t["id"], {}).get("executed_irreversible_action"),
                  "rejected_calls": audits.get(t["id"], {}).get("rejected_calls"),
                  "stated_text": development_stated[t["id"]]["text"],
                  "terminal_status": next((r["status"] for r in behavior_report["episodes"] if r["id"] == t["id"]), None),
                  "episode_file": next((r["episode_file"] for r in behavior_report["episodes"] if r["id"] == t["id"]), None),
                  "infrastructure_failure": audits.get(t["id"], {}).get("infrastructure_failure"),
                  "RI_gap_eligible": t["RI_gap_eligible"]} for i, t in enumerate(tasks)]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Verified descriptive joins for", len(summaries), "probe conditions and", behavior["eligible_RI_pairs"], "development R/I pairs")


if __name__ == "__main__":
    main()
