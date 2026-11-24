"""Run an explicitly invoked child command with the Mac memory, disk, and time guards."""

import argparse
import os
import signal
import subprocess
from pathlib import Path

from course_llm_audit import load_json, save
from run_mac_safer_behavior import CONFIG, MacBudget


def run(command, directory, seconds):
    directory.mkdir(parents=True, exist_ok=False)
    config = load_json(CONFIG)
    config["wall_seconds"] = seconds
    budget = MacBudget(config, directory)
    result = {"command": command, "limits": config, "initial_memory": budget.baseline}
    try:
        with budget, (directory / "stdout.log").open("x") as out, (directory / "stderr.log").open("x") as err:
            child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            try:
                while child.poll() is None:
                    budget.check()
                    try:
                        child.wait(timeout=config["sample_seconds"])
                    except subprocess.TimeoutExpired:
                        continue
            except BaseException:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    except ProcessLookupError:
                        child.wait()
                raise
            result["exit_code"] = child.returncode
    except Exception as error:
        result["error"] = str(error)
    finally:
        result["status"] = "PASS" if result.get("exit_code") == 0 and "error" not in result else "STOPPED"
        save(directory / "process.json", result)
    return result["status"] == "PASS"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("command", nargs="+")
    args = parser.parse_args()
    if not run(args.command, args.directory, args.seconds):
        raise SystemExit(1)
