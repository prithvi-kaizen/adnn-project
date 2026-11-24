"""Finish dependent C4 stages serially after the already-running selection job."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.feature_store import sha  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=14400)
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Explicit --run is required; no model is loaded by default")
    artifacts = ROOT / "artifacts/course_fast_track/native_mac_cpu_v1"
    output = artifacts / "controller"
    if output.exists():
        raise SystemExit("Preserve previous controller attempts")
    output.mkdir(parents=True)
    evidence = ROOT / "docs/evidence/course_phase4_20261115"
    results = ROOT / "results/course_fast_track/native_mac_cpu_v1"
    activation = ROOT / "artifacts/activations/course_native_mac_cpu_v1"
    features, models, probes = activation / "features", activation / "probes", results / "probes"
    length, length_models = results / "length", activation / "length"
    paths = [Path(__file__), ROOT / "scripts/run_course_length_control.py", ROOT / "scripts/run_course_probes.py",
             ROOT / "scripts/run_course_core.py", ROOT / "scripts/run_course_behavior.py",
             ROOT / "scripts/run_course_reference_policies.py", ROOT / "scripts/summarize_course_core.py",
             ROOT / "scripts/verify_course_core.py", ROOT / "scripts/verify_course_phase4.py"]
    paths += [p for directory in ("src/models", "src/data", "src/agent", "src/sandbox")
              for p in (ROOT / directory).glob("*.py")]
    paths += [ROOT / "src/probes/linear.py", ROOT / "src/probes/course.py",
              ROOT / "configs/course_execution_native_mac_cpu_v1.json", ROOT / "docs/COURSE_CORE_PROTOCOL.md"]
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    report = {"status": "WAITING_FOR_ALL_SELECTIONS", "started_unix": time.time(),
              "source_sha256": frozen, "stages": [], "sequential_native_processes": True,
              "new_model_downloads": False, "human_judgments_created": 0}
    save(output / "source_lock.json", report)

    def stage(name, program, *arguments):
        assert all(sha(ROOT / p) == value for p, value in frozen.items()), "A bound C4 source file changed"
        record = {"name": name, "started_unix": time.time(), "argv": [program, *map(str, arguments)]}
        report["stages"].append(record)
        save(output / "progress.json", report)
        print("Starting", name, flush=True)
        with (evidence / f"{name}.log").open("x") as log:
            completed = subprocess.run([sys.executable, str(ROOT / program), *map(str, arguments)], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT)
        record.update(returncode=completed.returncode, finished_unix=time.time())
        save(output / "progress.json", report)
        if completed.returncode:
            raise RuntimeError(f"{name} failed; preserve its outputs and log")
        print("Finished", name, flush=True)

    try:
        started = time.monotonic()
        last_notice = -60
        while True:
            lock_path = probes / "selection_lock.json"
            if lock_path.exists():
                try:
                    locked = json.loads(lock_path.read_text())
                except json.JSONDecodeError:
                    time.sleep(.2)
                    continue
                assert locked["status"] == "ALL_SELECTIONS_LOCKED_BEFORE_TEST_EVALUATION"
                assert locked["N_conditions"] == len(locked["receipts"]) == 180
                assert all(sha(ROOT / p) == value for p, value in locked["source_sha256"].items())
                break
            log = evidence / "probe_selection.log"
            if log.exists() and "Traceback (most recent call last)" in log.read_text()[-10000:]:
                raise RuntimeError("The selection job failed; do not unseal tests")
            if time.monotonic() - started > args.wait_seconds:
                raise TimeoutError("Selection wait limit reached; no test inference launched")
            if time.monotonic() - started - last_notice >= 60:
                count = len(list(models.glob("*.joblib")))
                print(f"Waiting for selection lock: {count}/180 saved conditions", flush=True)
                last_notice = time.monotonic() - started
            time.sleep(5)
        report["status"] = "RUNNING_SERIAL_DEPENDENT_STAGES"
        report["main_selection_lock_sha256"] = sha(lock_path)
        stage("length_selection", "scripts/run_course_length_control.py", "select", "--features", features,
              "--output", length, "--models", length_models)
        stage("probe_evaluation", "scripts/run_course_probes.py", "evaluate", "--features", features,
              "--models", models, "--results", probes)
        stage("length_evaluation", "scripts/run_course_length_control.py", "evaluate", "--features", features,
              "--output", length, "--models", length_models)
        stage("probe_verification", "scripts/verify_course_core.py", "probes", probes, "--models", models,
              "--output", evidence / "probes_verification.json")
        stage("length_verification", "scripts/verify_course_core.py", "length", length, "--main-probes", probes,
              "--output", evidence / "length_verification.json")
        stage("main_stated", "scripts/run_course_core.py", "stated", "--output", artifacts / "stated",
              "--selection-lock", lock_path, "--features-report", features / "report.json")
        stage("development_behavior", "scripts/run_course_behavior.py", "--output", artifacts / "behavior")
        stage("reference_policies", "scripts/run_course_reference_policies.py", "--output", artifacts / "references")
        stage("descriptive_summary", "scripts/summarize_course_core.py", "--probes", probes,
              "--stated", artifacts / "stated", "--behavior", artifacts / "behavior", "--length", length,
              "--models", models, "--output", results / "core_summary.json")
        stage("course_gate_c4", "scripts/verify_course_phase4.py", "--results", results, "--artifacts", artifacts,
              "--evidence", evidence, "--output", evidence / "gate_c4.json")
        report.update(status="PASS_SERIAL_C4_WORKFLOW", finished_unix=time.time(),
                      gate_sha256=sha(evidence / "gate_c4.json"))
        save(output / "report.json", report)
        print("C4 record verification passed; Phase V statistics and bonus remain separate", flush=True)
    except Exception:
        report.update(status="FAILED_OR_BLOCKED", error=traceback.format_exc(), finished_unix=time.time())
        save(output / "report.json", report)
        raise


if __name__ == "__main__":
    main()
