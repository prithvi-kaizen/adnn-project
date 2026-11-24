"""Build and audit the unfrozen sandbox expansion without reading model outcomes."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from audit_course_behavior_readiness import DATA, FROZEN, HISTORY, ROOT, read, read_rows, sha

sys.path.insert(0, str(ROOT))

from src.data.behavior_readiness import joint_exposure_graph  # noqa: E402
from src.data.course_expansion import MECHANISMS, PAIRS_PER_MECHANISM, SEED, candidate_records  # noqa: E402
from src.data.pilot import simulator_source_hash  # noqa: E402
from src.data.validation import validate_records  # noqa: E402

PRIOR_FIXTURE = Path("docs/evidence/course_improvement_v1/i2_development/behavior_readiness.json")


def horizon_counts(rows):
    return {str(horizon): dict(Counter(row["labels_by_horizon"][str(horizon)]["label"] for row in rows))
            for horizon in (1, 2, 3)}


def calculate():
    plan = read(Path("configs/course_improvement_v1.json"))
    assert sha(FROZEN) == plan["protected_baseline_sha256"][str(FROZEN)]
    frozen = read(FROZEN)
    assert sha(DATA) == frozen["files"][DATA.name]["sha256"]
    prior = read(PRIOR_FIXTURE)
    assert all(sha(Path(path)) == checksum for path, checksum in prior["source_sha256"].items())
    history = {path: read_rows(path) for path in HISTORY}
    baseline = history[DATA]
    rows = list(candidate_records(ROOT))
    assert len(rows) == 2 * PAIRS_PER_MECHANISM * len(MECHANISMS)
    combined = [*baseline, *rows]
    sandbox_source = simulator_source_hash(ROOT)
    validation = validate_records(combined, recompute=True, expected_source_hash=sandbox_source)
    assert validation["status"] == "PASS" and validation["unknown"] == 0, validation
    graph = joint_exposure_graph(
        [row for items in history.values() for row in items] + prior["development_fixture"], rows)
    by_group, by_mechanism = defaultdict(list), defaultdict(list)
    for row in rows:
        by_group[graph["candidate_assignments"][row["id"]]["group"]].append(row)
        by_mechanism[row["mechanism"]].append(row)
    groups = {}
    for name, members in by_group.items():
        statuses = {graph["candidate_assignments"][row["id"]]["quarantined"] for row in members}
        assert len(statuses) == 1
        groups[name] = {"quarantined": statuses.pop(), "N": len(members),
                        "mechanisms": sorted({row["mechanism"] for row in members}),
                        "horizon_counts": horizon_counts(members)}
    unlinked = [row for row in rows if not graph["candidate_assignments"][row["id"]]["quarantined"]]
    assert all(count == 2 for count in Counter(row["minimal_pair_id"] for row in unlinked).values())
    classes = {row["label"] for row in combined}
    all_classes_by_horizon = all(set(counts) == classes for counts in horizon_counts(unlinked).values())
    paths = [*HISTORY, FROZEN, PRIOR_FIXTURE, Path("configs/course_improvement_v1.json"),
             Path("scripts/build_course_sandbox_expansion.py"), Path("src/data/course_expansion.py"),
             Path("src/data/behavior_readiness.py"), Path("src/data/validation.py"),
             Path("src/sandbox/core.py"), Path("src/sandbox/labeling.py")]
    required_test = len(combined) // 5
    assert len(combined) % 5 == 0 and len(unlinked) < required_test
    summary = {"status": "PASS_SANDBOX_CANDIDATE_VALIDATION_ONLY", "phase_I2": "PARTIAL",
               "original_gate2": "PARTIAL", "freeze": False, "model_calls": 0, "human_annotations": 0,
               "cloud_annotations": 0, "certified_physical_rows": 0, "new_behavior_families": 0,
               "seed": SEED, "seeded_pairs_per_mechanism": PAIRS_PER_MECHANISM,
               "baseline_rows": len(baseline), "new_samples": len(rows), "combined_sandbox_rows": len(combined),
               "validation": validation, "combined_horizon_counts": horizon_counts(combined),
               "new_horizon_counts": horizon_counts(rows),
               "mechanisms": {name: {"N": len(items), "horizon_counts": horizon_counts(items),
                   "changed_property": MECHANISMS[name][0],
                   "groups": sorted({graph["candidate_assignments"][row["id"]]["group"] for row in items})}
                   for name, items in by_mechanism.items()},
               "group_graph": graph, "groups_containing_new_rows": groups,
               "unlinked_to_enumerated_history_rows": len(unlinked),
               "unlinked_to_enumerated_history_groups": sum(not group["quarantined"] for group in groups.values()),
               "unlinked_pool_contains_all_classes_at_every_horizon": all_classes_by_horizon,
               "rows_needed_for_20_percent_of_combined_pool": required_test,
               "exact_20_percent_untouched_test_possible_from_this_pool": False,
               "scope_of_exposure_review": "known_local_sandbox_history_only",
               "source_sha256": {str(path): sha(path) for path in paths}}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new candidate directory")
    rows, audit = calculate()
    args.output.mkdir(parents=True, exist_ok=False)
    datafile = args.output / "sandbox_additions.jsonl"
    datafile.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows))
    audit["candidate_file_sha256"] = sha(datafile.resolve())
    (args.output / "validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(audit["status"], audit["combined_sandbox_rows"], audit["unlinked_to_enumerated_history_rows"],
          audit["unlinked_to_enumerated_history_groups"], audit["combined_horizon_counts"])


if __name__ == "__main__":
    main()
