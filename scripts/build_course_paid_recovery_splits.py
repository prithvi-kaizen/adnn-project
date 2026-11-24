"""Build the paid-recovery expansion and verify prospective sandbox split feasibility."""

import argparse
import itertools
import json
from pathlib import Path

import scipy
from audit_course_behavior_readiness import DATA, HISTORY, ROOT, read, read_rows, sha
from build_course_sandbox_expansion import PRIOR_FIXTURE, horizon_counts

from src.data.behavior_readiness import joint_exposure_graph
from src.data.course_draft_split import allocate_draft_splits
from src.data.course_paid_recovery import DESIGNS, PAIRS_PER_DESIGN, SEED, paid_records
from src.data.pilot import simulator_source_hash
from src.data.validation import validate_records

EXPANSION = Path("data/course_improvement/v1/sandbox_expansion")


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def build():
    verified = read(EXPANSION / "validation.json")
    assert all(sha(Path(path)) == expected for path, expected in verified["source_sha256"].items())
    old_additions = EXPANSION / "sandbox_additions.jsonl"
    assert sha(old_additions) == verified["candidate_file_sha256"]
    assert verified["combined_sandbox_rows"] == 2040
    additions = list(paid_records(ROOT))
    earlier = read_rows(old_additions)
    baseline = read_rows(DATA)
    rows = [*baseline, *earlier, *additions]
    assert len(rows) == 2400 and len(additions) == 2 * len(DESIGNS) * PAIRS_PER_DESIGN
    history = [row for source in HISTORY for row in read_rows(source)] + read(PRIOR_FIXTURE)["development_fixture"]
    graph = joint_exposure_graph(history, rows)
    assignments = graph["candidate_assignments"]
    assert all(assignments[row["id"]]["quarantined"] for row in baseline)
    validation = validate_records(rows, recompute=True, expected_source_hash=simulator_source_hash(ROOT))
    if validation["status"] != "PASS" or validation["unknown"]:
        raise ValueError(f"Expansion failed full replay or quality gate: {validation}")
    qualified = [row for row in rows if not assignments[row["id"]]["quarantined"]]
    splits = allocate_draft_splits(rows, assignments)
    overlaps = {f"{left['seed']}-{right['seed']}":
                len(set(left["indices"]["test"]) & set(right["indices"]["test"]))
                for left, right in itertools.combinations(splits, 2)}
    paths = [*HISTORY, PRIOR_FIXTURE, old_additions, EXPANSION / "validation.json",
             Path(__file__).relative_to(ROOT), Path("scripts/audit_course_behavior_readiness.py"),
             Path("scripts/build_course_sandbox_expansion.py"), Path("src/data/behavior_readiness.py"),
             Path("src/data/course_draft_split.py"), Path("src/data/course_paid_recovery.py"),
             Path("src/data/validation.py"), Path("src/probes/course.py"), Path("src/data/splits.py")]
    audit = {"status": "PASS_DRAFT_SANDBOX_SPLIT_FEASIBILITY_ONLY", "I2": "PARTIAL", "original_gate2": "PARTIAL",
             "seed": SEED, "pairs_per_design": PAIRS_PER_DESIGN, "new_rows": len(additions),
             "new_designs": list(DESIGNS), "combined_rows": len(rows),
             "validator": validation, "all_horizon_counts": horizon_counts(rows),
             "certified_scienceworld_rows": 0, "annotation_responses_received": 0,
             "new_behavior_families_certified": 0, "new_model_calls": 0,
             "unlinked_candidate_rows": len(qualified),
             "unlinked_components": len({assignments[row["id"]]["group"] for row in qualified}),
             "unlinked_horizon_counts": horizon_counts(qualified),
             "exposure_definition": "no link in enumerated local history under conservative graph; not proof of external independence",
             "overlap_test_row_counts": overlaps, "overlapping_seeds_are_independent_replications": False,
             "all_splits_authorize_model_runs": False, "final_split_frozen": False,
             "scipy_version": scipy.__version__, "source_sha256": {str(path): sha(path) for path in paths}}
    return additions, assignments, splits, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh draft output directory")
    additions, assignments, splits, audit = build()
    args.output.mkdir(parents=True, exist_ok=False)
    additions_path = args.output / "paid_recovery_additions.jsonl"
    additions_path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in additions))
    components = args.output / "components.json"
    save(components, assignments)
    output = [additions_path, components]
    for split in splits:
        path = args.output / f"draft_seed_{split['seed']}.json"
        save(path, split)
        output.append(path)
    audit["output_sha256"] = {path.name: sha(path.resolve()) for path in output}
    save(args.output / "validation.json", audit)
    print(audit["status"], audit["combined_rows"], audit["unlinked_candidate_rows"],
          audit["unlinked_components"], audit["overlap_test_row_counts"])


if __name__ == "__main__":
    main()
