"""Prepare private next-state prediction inputs and measure label overlap."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_bytesized32_recoverability import serialize

SYSTEM = (
    "Predict one game-state transition. Return only a JSON object containing set, remove and order. "
    "In set, map each added or changed object ID to its complete new object record. "
    "In remove, list deleted object IDs. In order, give the new ordered ID list if it changes, or null otherwise. "
    "Also return allocator and score, using null for each of those fields when unchanged. "
    "Use string object IDs and leave unchanged object records out of set."
)


def indexed(rows):
    result = {str(row["uuid"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate object identity")
    return result


def prediction_target(current, future, score, next_score):
    before, after = indexed(current["objects"]), indexed(future["objects"])
    return {"set": {key: value for key, value in after.items() if before.get(key) != value},
            "remove": sorted(set(before) - set(after)), "order": list(after) if list(before) != list(after) else None,
            "allocator": future["max_UUID"] if current["max_UUID"] != future["max_UUID"] else None,
            "score": next_score if score != next_score else None}


def reconstruct(current, score, delta):
    objects = indexed(current["objects"])
    for key in delta["remove"]:
        del objects[key]
    objects.update(delta["set"])
    order = list(objects) if delta["order"] is None else delta["order"]
    if len(order) != len(objects) or set(order) != set(objects):
        raise ValueError("Delta has inconsistent object order")
    return {"objects": [objects[key] for key in order],
            "allocator": current["max_UUID"] if delta["allocator"] is None else delta["allocator"],
            "score": score if delta["score"] is None else delta["score"]}


def prepare(case):
    before, after = case["current_state"], case["next_state"]
    delta = prediction_target(before, after, case["current_score"], case["next_score"])
    assert reconstruct(before, case["current_score"], delta) == {
        "objects": after["objects"], "allocator": after["max_UUID"], "score": case["next_score"]}
    payload = {"task": before["taskDesc"], "objects": before["objects"], "allocator": before["max_UUID"],
               "score": case["current_score"], "action": case["action"]}
    user = serialize(payload)
    return {"id": case["id"], "game": case["game"], "verb": case["verb"], "label": case["label"],
            "prefix_depth": case["prefix_depth"], "messages": [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}], "gold_delta": delta,
            "changed_object_count": len(delta["set"]) + len(delta["remove"]),
            "input_object_count": len(before["objects"])}


def prepare_file(case_path, results, private_output=None):
    raw = case_path.read_bytes()
    items = [json.loads(line) for line in raw.splitlines()]
    ids = {row["id"] for row in items}
    if not items or len(items) != len(ids):
        raise ValueError("Nonempty unique cases required")
    distinct = {}
    for case in sorted(items, key=lambda row: row["id"]):
        key = (case["game"], case["causal_state_id"], serialize(case["semantic_action"]))
        if key in distinct and distinct[key]["label"] != case["label"]:
            raise ValueError("Semantic duplicate has conflicting labels")
        distinct.setdefault(key, case)
    prepared_by_id = {case["id"]: prepare(case) for case in items}
    inputs = [prepared_by_id[case["id"]] for case in distinct.values()]
    label_counts = Counter(row["label"] for row in inputs)
    strata = defaultdict(Counter)
    for row in inputs:
        strata[(row["game"], row["verb"])][row["label"]] += 1
    overlaps = {f"{game}:{verb}": dict(sorted(counts.items())) for (game, verb), counts in sorted(strata.items())}
    shared = {name: counts for name, counts in overlaps.items() if len(counts) > 1}
    matched_games = sorted({name.split(":")[0] for name in shared})
    report = {"scope": "diagnostic_cohort_input_and_design_audit", "input_sha256": hashlib.sha256(raw).hexdigest(),
              "archived_labeled_rows": len(items), "unique_semantic_cases": len(inputs),
              "semantic_duplicates_excluded": len(items) - len(inputs),
              "game_groups": len({row["game"] for row in inputs}), "label_counts": dict(label_counts),
              "game_verb_counts": overlaps, "game_verb_strata_with_both_labels": shared,
              "games_with_within_verb_overlap": matched_games, "all_gold_deltas_reconstruct": True,
              "stable_primary_cohort_frozen": False, "model_outputs_read": 0}
    if private_output:
        with private_output.open("x") as stream:
            for row in inputs:
                stream.write(serialize(row) + "\n")
        report["private_design_sha256"] = hashlib.sha256(private_output.read_bytes()).hexdigest()
    with results.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--private-output", type=Path)
    args = parser.parse_args()
    prepare_file(args.cases, args.receipt, args.private_output)
