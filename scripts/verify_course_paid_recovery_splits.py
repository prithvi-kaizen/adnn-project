"""Recount draft split support and audit cross-seed exposure from immutable records."""

import argparse
import itertools
from collections import Counter, defaultdict
from pathlib import Path

from audit_course_behavior_readiness import DATA, read, read_rows, sha
from build_course_paid_recovery_splits import EXPANSION, save

LABELS = {"reversible", "partially_reversible", "irreversible"}


def verify(directory):
    audit = read(directory / "validation.json")
    assert not audit["final_split_frozen"] and not audit["all_splits_authorize_model_runs"]
    assert all(sha(Path(path)) == expected for path, expected in audit["source_sha256"].items())
    assert all(sha((directory / name).resolve()) == expected for name, expected in audit["output_sha256"].items())
    rows = [row for path in (DATA, EXPANSION / "sandbox_additions.jsonl",
                             directory / "paid_recovery_additions.jsonl") for row in read_rows(path)]
    by_id = {row["id"]: row for row in rows}
    components = read(directory / "components.json")
    assert len(by_id) == len(rows) == 2400 and by_id.keys() == components.keys()
    pairs = defaultdict(set)
    for row in rows:
        if row["minimal_pair_id"]:
            pairs[row["minimal_pair_id"]].add(row["id"])
    assert len(pairs) == 1110 and all(len(pair) == 2 for pair in pairs.values())
    splits = [read(directory / f"draft_seed_{seed}.json") for seed in (1729, 2718, 31415)]
    checks = 0
    testing = {}
    for split in splits:
        assert not split["reuse_allowed_for_model_runs"]
        assigned = split["indices"]
        assert [len(assigned[name]) for name in ("train", "development", "test")] == [1440, 480, 480]
        membership = {key: name for name, members in assigned.items() for key in members}
        assert len(membership) == 2400 and membership.keys() == by_id.keys()
        group_sets = {name: {components[key]["group"] for key in members} for name, members in assigned.items()}
        assert all(not left & right for left, right in itertools.combinations(group_sets.values(), 2))
        assert all(len({membership[key] for key in pair}) == 1 for pair in pairs.values())
        assert len(group_sets["test"]) == 8
        assert not any(components[key]["quarantined"] for key in assigned["test"])
        assert set(assigned["test"]).isdisjoint(row["id"] for row in read_rows(DATA))
        for horizon in ("1", "2", "3"):
            truth = {key: row["labels_by_horizon"][horizon]["label"] for key, row in by_id.items()}
            for mode, labels in (("threeway", LABELS), ("binary", LABELS - {"partially_reversible"})):
                intact = set()
                for pair in pairs.values():
                    observed = {truth[key] for key in pair}
                    if len(observed) == 2 and observed <= labels:
                        intact.update(pair)
                for subset in ("all", "pairs"):
                    key = f"h{horizon}_{mode}_{subset}"
                    for part, ids in assigned.items():
                        counts = Counter(truth[item] for item in ids
                                         if truth[item] in labels and (subset == "all" or item in intact))
                        assert set(counts) == labels and dict(counts) == split["class_counts"][part][key]
                        checks += 1
        testing[split["seed"]] = set(assigned["test"])
    union = set.union(*testing.values())
    intersections = {f"{left}-{right}": len(testing[left] & testing[right])
                     for left, right in itertools.combinations(testing, 2)}
    assert intersections == audit["overlap_test_row_counts"]
    return {"status": "PASS_INDEPENDENT_DRAFT_SPLIT_RECOUNT", "class_support_recounts": checks,
            "suite_sizes": [len(suite) for suite in testing.values()],
            "unique_rows_in_any_draft_test": len(union),
            "union_of_test_components": len({components[key]["group"] for key in union}),
            "pairwise_test_overlap_rows": intersections,
            "per_seed_rows_in_other_test_sets": {str(seed): len(union - ids) for seed, ids in testing.items()},
            "proposed_prompt_or_steering_dev_exclusion": "all components in any draft test",
            "authorizes_inference": False, "joint_environment_gate_passed": False,
            "audit_sha256": sha((directory / "validation.json").resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.draft)
    save(args.receipt, result)
    print(result)


if __name__ == "__main__":
    main()
