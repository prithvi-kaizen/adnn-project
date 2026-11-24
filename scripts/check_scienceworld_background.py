"""Time-matched exploratory control; no primary reversibility labels."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_scienceworld_state import differences  # noqa: E402
from src.scienceworld_adapter import ScienceWorldReplay  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Preserve prior evidence; choose a new directory")
    args.output_dir.mkdir(parents=True)
    env = ScienceWorldReplay("boil", 0)
    active = ("open door to art studio", "close door to art studio")
    passive = ("wait1",)
    try:
        initial = env.replay(())
        a = env.replay(active)
        active_ticks = env.env.get_num_moves()
        b = env.replay(passive)
        passive_ticks = env.env.get_num_moves()
        aa, bb = env.replay(active), env.replay(passive)
        pairs = {"initial_active": (initial, a), "initial_passive": (initial, b), "active_passive": (a, b)}
        changes = {key: differences(json.loads(x.payload), json.loads(y.payload)) for key, (x, y) in pairs.items()}
        report = {"scope": "exploratory_environment_control_not_primary_labels",
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "proposal_sha256": hashlib.sha256((ROOT / "docs/SCIENCEWORLD_BACKGROUND_CONTROL_v2.md").read_bytes()).hexdigest(),
                  "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "task": "boil", "variation": 0, "simplifications": "",
                  "active_actions": active, "passive_actions": passive,
                  "engine_world_ticks": {"active": active_ticks, "passive": passive_ticks},
                  "time_match_verified": active_ticks == passive_ticks,
                  "python": platform.python_version(),
                  "scienceworld_version": importlib.metadata.version("scienceworld"),
                  "java_version": subprocess.run(["java", "-version"], capture_output=True, text=True).stderr,
                  "repeat_active_export_equal": a.payload == aa.payload,
                  "repeat_passive_export_equal": b.payload == bb.payload,
                  "exported_payload_hashes": {"initial": initial.digest(), "active": a.digest(), "passive": b.digest()},
                  "leaf_difference_counts": {key: len(value) for key, value in changes.items()},
                  "leaf_differences": changes, "state_export_complete": False,
                  "eligible_labels_produced": 0}
        for name, state in (("initial", initial), ("active", a), ("passive", b)):
            (args.output_dir / (name + ".json")).write_text(state.payload + "\n")
        (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "leaf_differences"}, indent=2))
        if not report["time_match_verified"]:
            raise SystemExit("Invalid timing control; preserve attempt and do not interpret")
    finally:
        env.close()


if __name__ == "__main__":
    main()
