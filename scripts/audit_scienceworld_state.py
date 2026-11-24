"""Record targeted inverse failure without mistaking it for exhaustive search."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scienceworld_adapter import ScienceWorldReplay  # noqa: E402


def differences(a, b, path=""):
    if type(a) is not type(b):
        return [(path, a, b)]
    if isinstance(a, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append((path + "/" + key, a.get(key), b.get(key)))
            else:
                out.extend(differences(a[key], b[key], path + "/" + key))
        return out
    return [] if a == b else [(path, a, b)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve prior evidence; choose a new output")
    env = ScienceWorldReplay("boil", 0)
    try:
        actions = ("open door to art studio", "close door to art studio")
        before, after = env.replay(()), env.replay(actions)
        interface = env.env.server.agentInterface().get()
        report = {"task": "boil", "variation": 0, "actions": list(actions),
                  "before_hash": before.digest(), "after_hash": after.digest(),
                  "exported_state_equal": before.payload == after.payload,
                  "changed_exported_fields": differences(json.loads(before.payload), json.loads(after.payload)),
                  "agent_interface_declared_fields": [str(f.getName()) for f in
                                                       interface.getClass().getDeclaredFields()],
                  "scope": "targeted_recovery_diagnostic_not_exhaustive_label", "export_complete": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
