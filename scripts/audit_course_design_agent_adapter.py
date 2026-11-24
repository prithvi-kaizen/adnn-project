"""Check all registered scripted routes through the unchanged agent shell."""

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_course_behavior_readiness import ROOT, read, sha

from src.agent.design_v1 import VERSION, run_scripted_design_episode
from src.sandbox import Action

DESIGN = Path("docs/evidence/course_improvement_v1/i2_behavior_design/verification.json")


def audit():
    source = read(DESIGN)
    assert source["status"] == "PASS_SCRIPTED_BEHAVIOR_FIXTURES_ONLY"
    assert all(sha(Path(path)) == expected for path, expected in source["source_sha256"].items())
    outcomes = {}
    counts = {}
    for task in source["tasks"]:
        key = task["id"]
        outcomes[key] = {}
        for name in ("direct", "confirmation_then_shortcut", "safer"):
            witness = source["receipts"][key][name]
            actions = [*([event["action"] for event in witness["events"]]), witness["terminal"]]
            script = [Action.make(action["tool"], **action["arguments"]).text() for action in actions]
            episode = run_scripted_design_episode(task, script)
            score = episode["score"]
            envelope = episode["transport_episode"]
            assert envelope["status"] == "finish" and len(envelope["trace"]) == witness["turns_including_finish"]
            assert score["goal_success"] == witness["goal_met_at_finish"] and score["goal_success"]
            assert envelope["final_hash"] == witness["final_state_hash"] and score["rejected_calls"] == 0
            assert [(record["before_hash"], record["oracle"]) for record in score["pre_action_oracle_records"]] == [
                (record["before_hash"], record["labels_by_horizon"]["3"]) for record in witness["events"]]
            if name == "direct":
                assert score["caution"] == 0 and score["shortcut_committed"]
                assert score["executed_irreversible_action"] == (task["expected_label_h3"] == "irreversible")
            elif name == "confirmation_then_shortcut":
                assert {"step": 1, "subtype": "simulated_confirmation"} in score["caution_events"]
            else:
                assert score["caution"] == 1 and not score["shortcut_committed"]
                assert not score["executed_irreversible_action"] and not score["unknown_action_labels"]
                assert "certified_safe_alternative" in {item["subtype"] for item in score["caution_events"]}
            counts.setdefault(name, Counter()).update(
                episodes=1, goal_success=int(score["goal_success"]), caution=score["caution"],
                any_irreversible=int(score["executed_irreversible_action"]),
                legacy_and_new_goal_differ=int(envelope["goal_met_at_termination"] != score["goal_success"]))
            outcomes[key][name] = episode
    paths = [DESIGN, Path("src/agent/design_v1.py"), Path("src/agent/loop.py"),
             Path("src/data/course_behavior_design.py"), Path(__file__).relative_to(ROOT)]
    return {"status": "PASS_SCRIPTED_GOAL_ADAPTER_ONLY", "version": VERSION,
            "I2": "PARTIAL", "original_gate2": "PARTIAL", "source_sha256": {str(path): sha(path) for path in paths},
            "episodes": 3 * len(source["tasks"]), "totals_by_scripted_policy": counts,
            "model_calls": 0, "holdout_episodes": 0, "authorizes_model_use": False, "raw_episode_records": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new receipt")
    result = audit()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(result["status"], result["totals_by_scripted_policy"])


if __name__ == "__main__":
    main()
