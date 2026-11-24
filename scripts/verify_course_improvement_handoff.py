"""Reproduce the extension's saved-record scope and outcomes without model dependencies."""

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.design_v1 import audit_design_episode  # noqa: E402
from src.agent.loop import parse_output  # noqa: E402
from src.sandbox import ToolError  # noqa: E402

EVIDENCE = Path("docs/evidence/course_improvement_v1")
FINAL = EVIDENCE / "final"
CORE = Path("results/course_fast_track/native_mac_cpu_v1")
MODEL = Path("artifacts/course_improvement/v1/behavior_design_model_diagnostic")
DRAFT = Path("data/course_improvement/v1/paid_recovery_split_drafts")
LAYERWISE = Path("results/course_improvement/v1/layerwise")


def read(path):
    return json.loads((ROOT / path).read_text())


def lines(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines()]


def sha(path):
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_label(text):
    try:
        value = json.loads(text, object_pairs_hook=tuple)
    except ValueError:
        return None
    if type(value) is tuple and len(value) == 1 and value[0][0] == "label":
        label = value[0][1]
        if label in ("reversible", "partially_reversible", "irreversible"):
            return label
    return None


def behavior_recount():
    design = read(EVIDENCE / "i2_behavior_design/verification.json")
    report = read(MODEL / "report.json")
    verification = read(EVIDENCE / "i2_behavior_model/verification.json")
    assert report["status"] == "DIAGNOSTIC_OBSERVED"
    assert report["holdout_calls"] == verification["holdout_calls"] == 0
    assert sha(MODEL / "report.json") == verification["report_sha256"]
    tasks = {row["id"]: row for row in design["tasks"]}
    assert len(tasks) == 6 and report["assigned_ids"] == list(tasks)
    assert [row["id"] for row in report["records"]] == list(tasks)
    responses = {row["request_id"]: row for row in lines(MODEL / "generation_responses.jsonl")}
    totals = Counter()
    by_side = defaultdict(Counter)
    pairs = defaultdict(dict)
    observed = {report["timing"]["synthetic_request_id"]}
    for row in report["records"]:
        task = tasks[row["id"]]
        episode_path = MODEL / row["episode"]
        assert sha(episode_path) == row["episode_sha256"]
        episode = read(episode_path)
        score = audit_design_episode(task, episode)
        assert score == row["score"]
        assert episode["status"] == "consecutive_invalid_abort"
        stated_id = row["stated_request_id"]
        assert stated_id not in observed
        observed.add(stated_id)
        stated = strict_label(responses[stated_id]["text"])
        assert stated == row["stated_label"] == "reversible"
        actual = task["expected_label_h3"]
        row_counts = Counter(N=1, correct_stated=int(stated == actual), goal=int(score["goal_success"]),
                             caution=score["caution"], any_irreversible=int(score["executed_irreversible_action"]),
                             invalid_abort=1)
        by_side[actual].update(row_counts)
        totals.update(row_counts)
        for event in episode["trace"]:
            request = event["generation"]["request_id"]
            assert request not in observed and responses[request] == event["generation"]
            observed.add(request)
            if event["status"] == "executed":
                totals["executed_calls"] += 1
            if event["status"] != "rejected":
                continue
            try:
                parse_output(event["raw_output"])
            except ToolError:
                totals["parse_rejected"] += 1
            else:
                totals["state_or_schema_rejected"] += 1
        pairs[task["pair_id"]][actual] = (stated == actual, score["caution"])
    assert observed == set(responses) and len(observed) == 23
    pair_outcomes = {}
    for key, sides in pairs.items():
        reversible, irreversible = (sides[label] for label in ("reversible", "irreversible"))
        knowledge = int(reversible[0] and irreversible[0])
        action = int(irreversible[1] > reversible[1])
        pair_outcomes[key] = {"K": knowledge, "A": action, "gap": knowledge - action}
    assert totals == Counter(N=6, correct_stated=3, goal=3, caution=0, any_irreversible=4, invalid_abort=6,
                             executed_calls=4, parse_rejected=8, state_or_schema_rejected=4)
    assert not any(value for row in pair_outcomes.values() for value in row.values())
    assert not (set(tasks) & set(read(CORE / "features_manifest.json")["ids"]))
    return {"totals": dict(totals), "by_initial_shortcut_class": dict(by_side), "paired_outcomes": pair_outcomes,
            "new_same_ID_RQ1_records": 0, "raw_calls_checked": len(observed), "original_model_results_replaced": False}


def build():
    readiness = read(FINAL / "i3_readiness.json")
    assert not readiness["live_allocation"] and readiness["new_downloads"] == 0
    assert datetime.fromisoformat(readiness["full_A100_test_only_forecast"]) > datetime.fromisoformat(
        readiness["operational_deadline_utc"])
    data = read(DRAFT / "validation.json")
    assert not data["final_split_frozen"] and data["combined_rows"] == 2400
    assert all(sha(DRAFT / name) == checksum for name, checksum in data["output_sha256"].items())
    review = read("docs/evidence/course_llm_annotation_v1/preparation.json")
    physical = read("docs/evidence/course_llm_annotation_v1/scienceworld_evidence_recheck.json")
    assert review["model_returns_received"] == review["human_annotations"] == physical["eligible_physical_labels"] == 0
    layer = read(LAYERWISE / "verification.json")
    assert sha(LAYERWISE / "statistics.json.gz") == layer["statistics_sha256"]
    with gzip.open(ROOT / LAYERWISE / "statistics.json.gz", "rt") as handle:
        layer_records = json.load(handle)["conditions"]
    assert len(layer_records) == 180 and sum(len(row["layers"]) for row in layer_records) == 4500
    selection = [row for row in read(CORE / "statistics.json")["conditions"]
                 if row["id"].endswith("_h3_binary_last_all_ordinary")]
    contrast = [row["heldout_minimal_pairs"]["primary_probe_minus_lexical"]["paired_difference"]["balanced_accuracy"]
                for row in selection]
    assert len(contrast) == 3 and all(row["estimate"] == 0 for row in contrast)
    steering = read("results/course_fast_track/native_first_decision_steering_v1/summary.json")
    assert steering["first_decision_cells"] == 102 and not steering["complete_full_RQ4"]
    return {
        "scope": "local_reduced_course_supplement_no_new_inference",
        "I3": {"status": "BLOCKED_NEW_NATIVE_STUDY", "pinned_cpu_files_verified": len(readiness["pinned_files_checked"]),
               "full_A100_forecast_IST": readiness["full_A100_test_only_forecast"]},
        "I4": {"status": "BLOCKED_NEW_CORE_PASS_SAVED_DEVELOPMENT_RECOUNT", "development": behavior_recount()},
        "I5": {"status": "PARTIAL_SAVED_STATISTICS_VERIFIED", "layer_records": 4500,
               "selectivity_intervals": layer["selectivity_intervals_recomputed_with_explicit_rows"],
               "matched_primary_balanced_accuracy_deltas": [row["estimate"] for row in contrast],
               "matched_primary_intervals": [row["CI95"] for row in contrast],
               "first_decision_steering_cells": steering["first_decision_cells"], "full_RQ4": "NOT_RUN", "RQ5": "NOT_RUN"},
        "draft_data": {"sandbox_candidates": data["combined_rows"], "paired_fraction": data["validator"]["minimal_pair_fraction"],
                       "unlinked_rows": data["unlinked_candidate_rows"], "unlinked_components": data["unlinked_components"],
                       "draft_test_rows": 480, "pairwise_test_overlap_rows": data["overlap_test_row_counts"],
                       "released_for_model_use": False},
        "outstanding": {"certified_ScienceWorld_rows": 0, "human_annotations": 0, "model_review_returns": 0,
                        "original_60_family_suite_complete": False, "prescribed_native_BF16_conditions_complete": 0},
        "external_submission": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-evidence", action="store_true")
    args = parser.parse_args()
    result = build()
    path = ROOT / FINAL / "i3_i5_evidence.json"
    if args.write_evidence:
        with path.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    else:
        assert result == read(path)
        pdf = read(FINAL / "pdf_qa.json")
        assert pdf["status"] == "PASS_TWO_PAGE_RENDER_AND_CONTENT_REVIEW"
        assert pdf["page_count"] == pdf["rendered_pages_visually_checked"] == 2
        assert sha(Path(pdf["pdf"])) == pdf["pdf_sha256"]
    print("PASS: pinned readiness facts, draft hashes, six model replays, and saved uncertainty records.")
    print("No inference ran; I3/I4 and the full I5 gate remain open.")


if __name__ == "__main__":
    main()
