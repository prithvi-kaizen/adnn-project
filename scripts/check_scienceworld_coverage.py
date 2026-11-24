"""Audit portal visibility and runtime field coverage, without changing labels."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scienceworld_adapter import ScienceWorldReplay  # noqa: E402
from src.scienceworld_adapter.instrumentation import coverage_snapshot  # noqa: E402


def main():
    from scienceworld.constants import JAR_PATH
    output = ROOT / "data/scienceworld/coverage_v1"
    if output.exists():
        raise SystemExit("Choose a new version before another attempt; preserve prior evidence")
    output.mkdir(parents=True)
    reports = []
    for task, variation in (("boil", 0), ("chemistry-mix", 0), ("freeze", 1)):
        env = ScienceWorldReplay(task, variation)
        try:
            base = env.replay(())
            initial = coverage_snapshot(env)
            action = next(a for a in env.legal_actions() if a.startswith("open door to "))
            env.replay((action,))
            after = coverage_snapshot(env)
            env.replay(())
            replayed = coverage_snapshot(env)
            for label, snapshot in (("initial", initial), ("after", after)):
                (output / f"{task}-{variation}-{label}.json").write_text(json.dumps(snapshot, indent=2) + "\n")
            portals = [o for o in initial["objects_including_portals"] if o["json_export"].get("propPortal")]
            after_map = {o["uuid"]: o for o in after["objects_including_portals"]}
            altered = [{"uuid": p["uuid"], "name": p["json_export"]["name"],
                        "before": p["json_export"]["propPortal"],
                        "after": after_map[p["uuid"]]["json_export"]["propPortal"]}
                       for p in portals if p["json_export"]["propPortal"] !=
                       after_map[p["uuid"]]["json_export"]["propPortal"]]
            reports.append({"task": task, "variation": variation, "action": action,
                            "complete": False, "eligible_labels": 0,
                            "exported_baseline_hash": base.digest(),
                            "augmented_initial_replay_equal": initial == replayed,
                            "flat_object_count": len(initial["objects_including_portals"]),
                            "portal_count": len(portals), "altered_portal_properties": altered,
                            "engine_iterations": [initial["engine_iteration"], after["engine_iteration"]]})
        finally:
            env.close()
    result = {"scope": "exploratory_state_export_coverage_not_final_labels",
              "scienceworld": importlib.metadata.version("scienceworld"), "python": platform.python_version(),
              "java": subprocess.run(["java", "-version"], capture_output=True, text=True).stderr,
              "jar_sha256": hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest(),
              "source_revision": "e8216d6044e8e39be9fcb185e3b2dfb602584b52", "records": reports,
              "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (Path(__file__).resolve(), ROOT / "src/scienceworld_adapter/instrumentation.py")}}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
