"""Scripted reference policies, explicitly separate from language-model results."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.course_metrics import audit_course_episode  # noqa: E402
from src.agent.loop import run_episode  # noqa: E402
from src.data.course import scenario_from_task  # noqa: E402
from src.models.feature_store import sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new reference-policy directory")
    args.output.mkdir(parents=True)
    tasks_path = ROOT / "data/course_fast_track/v2/development_tasks.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    records = []
    for task in tasks:
        for policy in ("direct_shortcut", "confirmation_then_goal", "always_refuse"):
            if policy == "direct_shortcut":
                actions = [json.loads(task["action_text"])]
            elif policy == "confirmation_then_goal":
                actions = task["cautious_goal_witness"]["actions"]
            else:
                actions = [{"tool": "refuse", "arguments": {"reason": "Fixed reference policy refuses every task."}}]
            sequence = iter(actions + [{"tool": "finish", "arguments": {"reason": "Fixed reference sequence ended."}}])
            episode = run_episode(scenario_from_task(task), lambda messages: {"text": json.dumps(next(sequence))})
            audit = audit_course_episode(task, episode)
            filename = f"{task['id']}_{policy}.json"
            record = {"policy": policy, "is_model_result": False, "is_scripted_reference": True,
                      "task_id": task["id"], "episode": episode, "audit": audit}
            (args.output / filename).write_text(json.dumps(record, indent=2) + "\n")
            records.append({"task_id": task["id"], "policy": policy, "file": filename,
                            "sha256": sha(args.output / filename), "audit": audit})
    report = {"status": "PASS_SCRIPTED_REFERENCE_REPLAY", "is_model_result": False,
              "human_judgments_created": 0, "tasks_sha256": sha(tasks_path), "assigned_versions": 24,
              "N_reference_episodes": len(records), "records": records,
              "knowledge_K_assigned_to_reference_policies": False,
              "confirmation_implies_recoverability": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Replayed", len(records), "scripted reference episodes; zero model judgments generated")


if __name__ == "__main__":
    main()
