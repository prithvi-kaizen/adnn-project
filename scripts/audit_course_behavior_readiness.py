"""Verify safer routes and quarantine exposed families without opening new holdouts."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.loop import goal_met  # noqa: E402
from src.data.behavior_readiness import certify_safe_route, joint_exposure_graph  # noqa: E402
from src.data.course import scenario_from_task  # noqa: E402
from src.sandbox.core import transition  # noqa: E402
from src.sandbox.labeling import label_action  # noqa: E402

POOL = Path("data/course_fast_track/v2")
FROZEN = POOL / "freeze_manifest.json"
DATA = POOL / "sandbox_pool.jsonl"
TASKS = POOL / "development_tasks.jsonl"
HISTORY = (
    Path("data/pilot/sandbox_pairs.jsonl"), Path("data/pilot/v2/sandbox_pairs.jsonl"),
    Path("data/pilot/v3/sandbox_pairs.jsonl"), Path("data/candidates/v4/sandbox_pairs.jsonl"),
    Path("data/candidates/v5/sandbox_pairs.jsonl"), DATA, TASKS,
)


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def read_rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines()]


def file_edit_pair(tasks):
    """A two-version R/I fixture from one exposed edit family, with no retained history."""
    parents = [task for task in tasks if task["pair_id"] == "pilot-edit-0"]
    assert len(parents) == 2
    output = []
    for task in parents:
        scenario = scenario_from_task(task)
        state = replace(scenario.state, history={})
        assert not goal_met(scenario, state)
        labels = {str(horizon): label_action(state, scenario.action, horizon).object() for horizon in (1, 2, 3)}
        direct, _ = transition(state, scenario.action)
        assert goal_met(scenario, direct)
        output.append({"id": f"route-fixture-no-history-{labels['3']['label']}",
                       "environment": "sandbox", "pair_id": "route-fixture-no-history",
                       "parent_scenario_id": task["id"], "template_id": task["template_id"],
                       "domain": task["domain"], "state_text": state.text(), "state_hash": state.digest(),
                       "goal": task["goal"], "action_text": task["action_text"],
                       "labels_by_horizon": labels, "label": labels["3"]["label"],
                       "scope": "exposed_development_fixture_not_heldout_data"})
    before, after = (json.loads(row["state_text"]) for row in output)
    assert [key for key in before if before[key] != after[key]] == ["backups"]
    assert output[0]["action_text"] == output[1]["action_text"] and output[0]["goal"] == output[1]["goal"]
    assert {row["label"] for row in output} == {"irreversible", "reversible"}
    return output


def calculate():
    config = read(Path("configs/course_improvement_v1.json"))
    assert sha(FROZEN) == config["protected_baseline_sha256"][str(FROZEN)]
    frozen = read(FROZEN)
    for path in (DATA, TASKS):
        assert sha(path) == frozen["files"][path.name]["sha256"], path
    history = {path: read_rows(path) for path in HISTORY}
    tasks = history[TASKS]
    assert len(tasks) == 24
    exposures = sorted(path.relative_to(ROOT) for path in (ROOT / "artifacts/course_fast_track")
                       .glob("**/pilot_exposures.jsonl"))
    exposures.append(Path("artifacts/exploratory/ollama_diagnostic_v2/requests.jsonl"))
    known_ids = defaultdict(set)
    for records in history.values():
        for row in records:
            assert hashlib.sha256(row["state_text"].encode()).hexdigest() == row["state_hash"], row["id"]
            known_ids[row["id"]].add(row["state_hash"])
    exposure_counts = {}
    non_scenario_canaries = 0
    for path in exposures:
        exposed = read_rows(path)
        for row in exposed:
            if (path.name == "requests.jsonl" and row.get("scenario_id") is None and
                    row.get("condition") in {"canary_raw", "canary_chat"}):
                non_scenario_canaries += 1
                continue
            key = row.get("scenario_id") or row["id"]
            assert key in known_ids, (path, key)
            if row.get("state_hash"):
                assert row["state_hash"] in known_ids[key], (path, key)
        exposure_counts[str(path)] = sum(bool(row.get("scenario_id") or row.get("id")) for row in exposed)
    fixture = file_edit_pair(tasks)
    graph = joint_exposure_graph([row for records in history.values() for row in records], fixture)
    assert all(item["quarantined"] for item in graph["candidate_assignments"].values())
    routes = {}
    for row in [*tasks, *fixture]:
        scenario = scenario_from_task(row)
        assert label_action(scenario.state, scenario.action, 3).label == row["label"]
        direct, _ = transition(scenario.state, scenario.action)
        assert goal_met(scenario, direct)
        routes[row["id"]] = {"candidate_label_h3": row["label"], "direct_goal": True,
                             **certify_safe_route(scenario)}
    paired = defaultdict(list)
    for task in tasks:
        paired[task["pair_id"]].append(task)
    ri_safe_pairs = [pair for pair, members in sorted(paired.items())
                     if {row["label"] for row in members} == {"reversible", "irreversible"} and
                     all(routes[row["id"]]["status"] == "VERIFIED_ALL_STEPS_REVERSIBLE_ROUTE" for row in members)]
    assert all(routes[row["id"]]["status"] == "VERIFIED_ALL_STEPS_REVERSIBLE_ROUTE" for row in fixture)
    paths = [*HISTORY, *exposures, FROZEN, Path("configs/course_improvement_v1.json"),
             Path("scripts/audit_course_behavior_readiness.py"), Path("src/data/behavior_readiness.py"),
             Path("src/data/splits.py"), Path("src/sandbox/core.py"), Path("src/sandbox/labeling.py"),
             Path("src/agent/loop.py")]
    return {"status": "PASS_DEVELOPMENT_ROUTE_AND_EXPOSURE_CHECK_ONLY", "gate_I2": "PARTIAL",
            "original_gate2": "PARTIAL", "model_calls": 0, "new_holdout_rows": 0,
            "new_certified_scienceworld_rows": 0, "completed_final_paired_behavior_families": 0,
            "history_rows_by_source": {str(path): len(records) for path, records in history.items()},
            "prior_exposure_records_joined": exposure_counts,
            "non_scenario_canaries_excluded": non_scenario_canaries,
            "all_frozen_course_pool_rows_excluded_from_untouched_claims": True,
            "coverage": "enumerated_local_sandbox_sources_only_not_external_session_history",
            "exposure_graph": graph, "development_versions": len(tasks), "development_pair_instances": len(paired),
            "route_status_counts_existing": dict(Counter(routes[row["id"]]["status"] for row in tasks)),
            "existing_RI_safe_pair_ids": ri_safe_pairs,
            "existing_RI_safe_pair_templates": sorted({paired[pair][0]["template_id"] for pair in ri_safe_pairs}),
            "development_fixture": fixture, "routes": routes,
            "source_sha256": {str(path): sha(path) for path in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh evidence path")
    result = calculate()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(result["status"], result["route_status_counts_existing"], result["existing_RI_safe_pair_templates"])


if __name__ == "__main__":
    main()
