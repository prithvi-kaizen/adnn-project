"""Certify distinct behavior design fixtures and screen their exposure links."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from audit_course_behavior_readiness import HISTORY, ROOT, read, read_rows, sha
from build_course_sandbox_expansion import PRIOR_FIXTURE

from src.data.behavior_readiness import joint_exposure_graph
from src.data.course_behavior_design import build_pairs, certify

DRAFT = Path("data/course_improvement/v1/paid_recovery_split_drafts")
ADDITIONS = Path("data/course_improvement/v1/sandbox_expansion/sandbox_additions.jsonl")


def audit():
    draft = read(DRAFT / "validation.json")
    assert not draft["final_split_frozen"]
    assert all(sha(DRAFT / name) == expected for name, expected in draft["output_sha256"].items())
    assert all(sha(Path(path)) == expected for path, expected in draft["source_sha256"].items())
    splits = [DRAFT / f"draft_seed_{seed}.json" for seed in (1729, 2718, 31415)]
    reserved = {key for split in splits for key in read(split)["indices"]["test"]}
    draft_rows = [row for path in (ADDITIONS, DRAFT / "paid_recovery_additions.jsonl")
                  for row in read_rows(path) if row["id"] in reserved]
    assert len(reserved) == len(draft_rows) == 510
    history = [row for path in HISTORY for row in read_rows(path)] + read(PRIOR_FIXTURE)["development_fixture"]
    tasks = build_pairs()
    historical_graph = joint_exposure_graph(history, tasks)
    protected_graph = joint_exposure_graph([*history, *draft_rows], tasks)
    grouped = defaultdict(list)
    receipts = {}
    for task in tasks:
        grouped[task["pair_id"]].append(task)
        receipts[task["id"]] = certify(task)
    family_checks = {}
    for key, members in grouped.items():
        assert len(members) == 2
        first, second = members
        assert all(first[name] == second[name] for name in ("goal", "objective", "action_text", "alternate"))
        original = historical_graph["candidate_assignments"]
        protected = protected_graph["candidate_assignments"]
        assert original[first["id"]] == original[second["id"]]
        assert protected[first["id"]] == protected[second["id"]]
        family_checks[key] = {"versions": [first["id"], second["id"]],
                              "historical_link": original[first["id"]]["quarantined"],
                              "blocked_by_history_or_reserved_probe_test": protected[first["id"]]["quarantined"],
                              "screen_group": protected[first["id"]]["group"],
                              "goal_route_and_alternative_verified": True}
    paths = [*HISTORY, PRIOR_FIXTURE, ADDITIONS, DRAFT / "paid_recovery_additions.jsonl",
             DRAFT / "validation.json", *splits, Path("src/data/course_behavior_design.py"),
             Path("src/data/behavior_readiness.py"), Path("src/agent/loop.py"),
             Path("src/sandbox/core.py"), Path("src/sandbox/labeling.py"), Path(__file__).relative_to(ROOT)]
    return {"status": "PASS_SCRIPTED_BEHAVIOR_FIXTURES_ONLY", "original_gate2": "PARTIAL",
            "development_families": len(grouped), "development_versions": len(tasks),
            "replayed_scripted_routes": 3 * len(tasks), "model_calls": 0,
            "new_holdout_families_frozen": 0, "protected_probe_test_rows": len(reserved),
            "requires_versioned_goal_adapter_for_model_use": True,
            "family_screen": family_checks,
            "unlinked_candidate_components": len({item["screen_group"] for item in family_checks.values()
                                                  if not item["blocked_by_history_or_reserved_probe_test"]}),
            "tasks": tasks, "receipts": receipts, "source_sha256": {str(path): sha(path) for path in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh evidence path")
    result = audit()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(result["status"], result["development_families"], result["family_screen"])


if __name__ == "__main__":
    main()
