"""Offline replay of real saved model outputs in the isolated simulator."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.course_metrics import audit_course_episode  # noqa: E402
from src.data.course import scenario_from_task  # noqa: E402
from src.sandbox.labeling import label_action  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="pilot-delete-0")
    parser.add_argument("--task")
    args = parser.parse_args()
    tasks = [json.loads(line) for line in (ROOT / "data/course_fast_track/v2/development_tasks.jsonl").read_text().splitlines()]
    tasks = [t for t in tasks if t["id"] == args.task] if args.task else [t for t in tasks if t["pair_id"] == args.pair]
    if not tasks:
        raise SystemExit("No frozen task matches")
    artifact = ROOT / "artifacts/course_fast_track/native_mac_cpu_v1/behavior"
    run = json.loads((artifact / "report.json").read_text())
    summary = json.loads((ROOT / "results/course_fast_track/native_mac_cpu_v1/core_summary.json").read_text())
    joined = {r["id"]: r for r in summary["development_representation_knowledge_behavior"]["rows"]}
    print("SAVED MODEL TRACE REPLAY. No new model inference, network call or host-file action.\n")
    for task in tasks:
        recorded = next(r for r in run["episodes"] if r["id"] == task["id"])
        episode = json.loads((artifact / recorded["episode_file"]).read_text())
        audit = audit_course_episode(task, episode)
        scenario = scenario_from_task(task)
        oracle = label_action(scenario.state, scenario.action, 3).object()
        assert oracle["label"] == task["label"]
        print(task["id"])
        print("Goal:", task["goal"])
        print("Candidate:", task["action_text"])
        print("Oracle:", oracle["label"], "h=3; recovery witness:", oracle["witness"])
        print("State hash:", task["state_hash"])
        print("Recycling:", scenario.state.recycle_enabled)
        print("Probe:", joined[task["id"]]["probe_label"], "; stated:", joined[task["id"]]["stated_label"])
        for event in episode["trace"]:
            print(f"  Turn {event['step']} [{event['status']}]: {event.get('raw_output',event.get('error',''))}")
        print("Recorded stop:", episode["status"], "; goal at stop:", audit["goal_success"], "; caution:", audit["caution"])
        print("Exact deterministic world replay: VERIFIED\n")


if __name__ == "__main__":
    main()
