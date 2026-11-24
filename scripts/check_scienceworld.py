"""Small pinned-JAR replay/search diagnostic; no eligible physical labels."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scienceworld_adapter import ScienceWorldReplay, search_recovery  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from scienceworld.constants import JAR_PATH
    if args.output_dir.exists():
        raise SystemExit("Preserve prior diagnostic evidence; choose a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, variation in [("boil", 0), ("chemistry-mix", 0), ("freeze", 1)]:
        env = ScienceWorldReplay(task, variation)
        try:
            initial = env.replay(())
            available = env.legal_actions()
            action = next((a for a in available if a.startswith("open door to ")), "wait1")
            prefix = (action,)
            a = env.replay(prefix)
            b = env.replay(prefix)
            c = env.replay(())
            record = {"task": task, "variation": variation, "candidate_action": action,
                      "initial_hash": initial.digest(), "replayed_hashes": [a.digest(), b.digest()],
                      "repeat_replay_equal": a.payload == b.payload,
                      "reset_equal": initial.payload == c.payload,
                      "state_export_complete": False,
                      "initial_legal_action_count": len(env.legal_actions()), "searches": []}
            for h in (1, 2, 3):
                record["searches"].append(search_recovery(
                    env, (), action, h, max_nodes=12, max_seconds=5,
                    diagnostic_only=True,
                ))
            (args.output_dir / f"{task}-{variation}-initial.json").write_text(initial.payload + "\n")
            (args.output_dir / f"{task}-{variation}-after.json").write_text(a.payload + "\n")
            rows.append(record)
        finally:
            env.close()
    report = {"scope": "runtime_diagnostic_not_eligible_labeled_dataset", "python": platform.python_version(),
              "scienceworld_distribution": importlib.metadata.version("scienceworld"),
              "source_revision": "e8216d6044e8e39be9fcb185e3b2dfb602584b52",
              "jar_sha256": hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest(),
              "java_version": subprocess.run(["java", "-version"], capture_output=True, text=True).stderr,
              "records": rows, "eligible_scienceworld_labels": 0}
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
