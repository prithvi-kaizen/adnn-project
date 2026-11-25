"""Compare veto and simulator-planned recovery on fixed, saved model traces."""

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.course_metrics import audit_course_episode  # noqa: E402
from src.agent.loop import goal_met  # noqa: E402
from src.data.behavior_readiness import certify_safe_route  # noqa: E402
from src.data.course import scenario_from_task  # noqa: E402
from src.sandbox import State  # noqa: E402

TASKS = ROOT / "data/course_fast_track/v2/development_tasks.jsonl"
BASELINE = ROOT / "artifacts/course_fast_track/native_mac_cpu_v1/behavior/report.json"


def summarize_episode(task, episode):
    audit = audit_course_episode(task, episode)
    if audit["unknown_action_labels"] or audit["infrastructure_failure"]:
        raise ValueError(f"Baseline contains an unresolved episode: {task['id']}")
    baseline = {
        "goal": audit["goal_success"],
        "irreversible": audit["executed_irreversible_action"],
    }
    row = {"id": task["id"], "baseline": baseline, "veto": baseline, "veto_with_planner": baseline}
    first = next((item for item in audit["pre_action_oracle_records"]
                  if item["oracle"]["label"] == "irreversible"), None)
    if first is None:
        return row

    event = next(event for event in episode["trace"] if event["step"] == first["step"])
    state = State.from_object(event["before"])
    if state.digest() != first["before_hash"] or event["action"] != first["action"]:
        raise ValueError(f"Veto point does not match the audited trace: {task['id']}")
    scenario = replace(scenario_from_task(task), state=state)
    veto_goal = goal_met(scenario, state)
    remaining = episode["max_steps"] - event["step"] + 1
    if veto_goal:
        plan = {"status": "GOAL_MET_AT_VETO"}
    elif remaining < 2:
        plan = {"status": "NO_TURNS_FOR_ACTION_AND_FINISH"}
    else:
        plan = certify_safe_route(scenario, max_turns=remaining)

    return {
        **row,
        "veto": {"goal": veto_goal, "irreversible": False},
        "veto_with_planner": {
            "goal": veto_goal or plan["status"] == "VERIFIED_ALL_STEPS_REVERSIBLE_ROUTE",
            "irreversible": False,
        },
        "intervention": {
            "step": event["step"],
            "action": event["action"],
            "initial_hash": state.digest(),
            "remaining_turns": remaining,
            "planner": plan,
        },
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    tasks = {task["id"]: task for task in map(json.loads, TASKS.read_text().splitlines())}
    baseline = json.loads(BASELINE.read_text())
    records = baseline["episodes"]
    if (baseline["unobserved_ids"] or len(records) != 24 or
            {record["id"] for record in records} != set(baseline["assigned_ids"]) or
            len(set(baseline["assigned_ids"])) != 24):
        raise ValueError("The fixed development cohort is incomplete")
    rows = []
    for record in records:
        path = BASELINE.parent / record["episode_file"]
        episode = json.loads(path.read_text())
        row = summarize_episode(tasks[record["id"]], episode)
        rows.append({**row, "baseline_trace_sha256": sha256(path)})
    return {
        "scope": "recorded_native_qwen_development_prefixes_with_scripted_sandbox_intervention",
        "model_inference": False,
        "task_manifest_sha256": sha256(TASKS),
        "baseline_index_sha256": sha256(BASELINE),
        "N": len(rows),
        "triggered": sum("intervention" in row for row in rows),
        "verified_routes": sum(
            row.get("intervention", {}).get("planner", {}).get("status") == "VERIFIED_ALL_STEPS_REVERSIBLE_ROUTE"
            for row in rows
        ),
        "outcomes": {
            arm: {endpoint: sum(row[arm][endpoint] for row in rows) for endpoint in ("goal", "irreversible")}
            for arm in ("baseline", "veto", "veto_with_planner")
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = run()
    with args.output.open("x") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    print(json.dumps({key: report[key] for key in ("N", "triggered", "verified_routes", "outcomes")}))


if __name__ == "__main__":
    main()
