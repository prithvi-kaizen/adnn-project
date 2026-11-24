"""Replay saved mail behavior, distinguishing the model from scripted references."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.design_v1 import audit_design_episode  # noqa: E402

EVIDENCE = ROOT / "docs/evidence/course_improvement_v1"
MODEL = ROOT / "artifacts/course_improvement/v1/behavior_design_model_diagnostic"
CASE = "behavior-v1-queue_while_preserving_notice-irreversible"


def read(path):
    return json.loads(path.read_text())


def main():
    fixture = next(row for row in read(EVIDENCE / "i2_behavior_design/verification.json")["tasks"]
                   if row["id"] == CASE)
    scripted = read(EVIDENCE / "i2_behavior_adapter/verification.json")["raw_episode_records"][CASE]
    attempts = [
        ("Saved model episode", read(MODEL / f"{CASE}.json")),
        ("Scripted confirmation", scripted["confirmation_then_shortcut"]["transport_episode"]),
        ("Scripted reversible route", scripted["safer"]["transport_episode"]),
    ]
    print("Saved replay only. No model is loaded or contacted.")
    print(fixture["goal"])
    for label, episode in attempts:
        audit = audit_design_episode(fixture, episode)
        support_delivered = "support" in episode["final_state"]["delivered_mail"]
        print(f"{label}: goal={audit['goal_success']}, irreversible={audit['executed_irreversible_action']}, "
              f"support_delivered={support_delivered}, rejected={audit['rejected_calls']}")


if __name__ == "__main__":
    main()
