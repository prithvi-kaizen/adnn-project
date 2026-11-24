"""Run the fixed cached-model engineering protocol, never the BF16 study."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agent.loop import initial_messages, parse_output, run_episode, verify_episode  # noqa: E402
from src.data.pilot import scenarios  # noqa: E402
from src.models.ollama_dev import (  # noqa: E402
    MODELS,
    OPTIONS,
    OllamaDevelopmentClient,
    ResourceMonitor,
    append_jsonl,
    render,
    resource_sample,
    utc,
)
from src.sandbox import legal_actions  # noqa: E402


def parser_diagnostic(scenario, raw):
    result = {"json_valid": False, "grammar_valid": False, "legal_action": False}
    try:
        json.loads(raw)
        result["json_valid"] = True
        action = parse_output(raw)
        result["grammar_valid"] = True
        result["legal_action"] = action.tool in {"finish", "refuse"} or action in legal_actions(scenario.state)
    except (ValueError, TypeError) as error:
        result["error_type"] = type(error).__name__
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=("cold", "suite"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Preserve every prior attempt; choose a new output directory")
    args.output_dir.mkdir(parents=True)
    all_scenarios = scenarios()
    fixtures = [{"id": s.scenario_id, "pair_id": s.pair_id, "template_id": s.template_id,
                 "domain": s.domain, "goal": s.goal, "state": s.state.object(),
                 "candidate_action": s.action.object()} for s in all_scenarios]
    fixture_payload = json.dumps(fixtures, sort_keys=True)
    (args.output_dir / "fixtures.json").write_text(fixture_payload + "\n")
    sources = sorted(list((ROOT / "src/sandbox").glob("*.py")) +
                     list((ROOT / "src/agent").glob("*.py")) +
                     [ROOT / "src/models/ollama_dev.py", ROOT / "src/data/pilot.py", Path(__file__).resolve(),
                      ROOT / "docs/EXPLORATORY_OLLAMA_PROTOCOL.md"])
    with zipfile.ZipFile(args.output_dir / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(source, str(source.relative_to(ROOT)))
    metadata = {"started_at_utc": utc(), "scope": "exploratory_engineering_only_not_final_RQ_claims",
                "model_key": args.model, "stage": args.stage, "options": OPTIONS,
                "fixture_sha256": hashlib.sha256(fixture_payload.encode()).hexdigest(),
                "fixture_scope": "all_12_pilot_pairs_excluded_from_final_splits",
                "source_snapshot_sha256": hashlib.sha256((args.output_dir / "source_snapshot.zip").read_bytes()).hexdigest(),
                "source_commit_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
                "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "python": platform.python_version(), "platform": platform.platform(),
                "physical_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
                "weights_downloaded": False, "final_BF16_protocol_changed": False}
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    client = None
    started = time.monotonic()
    try:
        client = OllamaDevelopmentClient(args.model, args.output_dir)
        metadata["preflight"] = client.preflight()
        metadata["maximum_initial_prompt_byte_bound"] = max(len(render(initial_messages(s)).encode()) + 130
                                                             for s in all_scenarios)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        with ResourceMonitor(args.output_dir / "resources.jsonl"):
            if args.stage == "cold":
                response = client.generate(initial_messages(all_scenarios[0]), label="cold_request")
                outcome = {"generation": response, "parser": parser_diagnostic(all_scenarios[0], response["text"]),
                           "post_request_memory": client.guard(), "scope": "cold_engineering_diagnostic"}
                (args.output_dir / "cold.json").write_text(json.dumps(outcome, indent=2) + "\n")
                append_jsonl(args.output_dir / "resources.jsonl", resource_sample())
            else:
                micro = all_scenarios[:8] if args.model == "1b" else [all_scenarios[i] for i in (0, 3, 4, 7)]
                for scenario in micro:
                    for condition, repetition in (("plain", 1), ("plain", 2), ("schema", 1)):
                        row = {"scenario_id": scenario.scenario_id, "pair_id": scenario.pair_id,
                               "template_id": scenario.template_id, "condition": condition, "repetition": repetition}
                        try:
                            response = client.generate(initial_messages(scenario), structured=condition == "schema",
                                                       label="microbenchmark")
                            row.update(status="completed", generation=response,
                                       parser=parser_diagnostic(scenario, response["text"]))
                        except Exception as error:
                            row.update(status="infrastructure_failure", error_type=type(error).__name__, error=str(error))
                        append_jsonl(args.output_dir / "micro.jsonl", row)
                        print(json.dumps({"progress": "micro", **{k: row[k] for k in
                                                                  ("scenario_id", "condition", "repetition", "status")}}), flush=True)
                episode_scenarios = all_scenarios if args.model == "1b" else all_scenarios[:8]
                assigned = [(s, 1) for s in episode_scenarios] + [(all_scenarios[i], 2) for i in (0, 4)]
                for number, (scenario, repetition) in enumerate(assigned, 1):
                    trace_path = args.output_dir / f"episode_{number:03d}_events.jsonl"
                    result = run_episode(scenario, lambda messages: client.generate(messages, label="agent_episode"),
                                         record_event=lambda event: append_jsonl(trace_path, event))
                    result["repetition"] = repetition
                    verify_episode(result)
                    path = args.output_dir / f"episode_{number:03d}.json"
                    path.write_text(json.dumps(result, indent=2) + "\n")
                    print(json.dumps({"progress": "episode", "number": number, "scenario": scenario.scenario_id,
                                      "status": result["status"], "steps": result["steps"],
                                      "goal_met": result["goal_met_at_termination"]}), flush=True)
                append_jsonl(args.output_dir / "resources.jsonl", resource_sample())
        # Digest verification is metadata-only, not another inference request.
        from src.models.ollama_dev import api
        selected = next(m for m in api("/api/tags")["models"] if m["name"] == client.model)
        if selected["digest"] != client.digest:
            raise RuntimeError("Model tag changed during run")
        metadata["status"] = "completed"
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__, error=str(error))
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error)}), flush=True)
    finally:
        if client is not None:
            try:
                metadata["residents_after_unload"] = client.unload()
            except Exception as error:
                metadata["unload_error"] = str(error)
        metadata.update(ended_at_utc=utc(), wall_seconds=time.monotonic() - started)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if metadata["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
