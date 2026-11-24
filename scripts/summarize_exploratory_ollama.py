"""Recompute exploratory engineering diagnostics from preserved raw records."""

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_exploratory_ollama import parser_diagnostic  # noqa: E402
from src.agent.loop import trace_signature, verify_episode  # noqa: E402
from src.data.pilot import scenarios  # noqa: E402


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def rate(rows, field):
    groups = defaultdict(list)
    for r in rows:
        groups[r["template_id"]].append(int(bool(r.get("parser", {}).get(field))))
    names = sorted(groups)
    n = len(rows)
    rng = random.Random(9001)
    sampled = []
    for _ in range(1000):
        values = [v for name in rng.choices(names, k=len(names)) for v in groups[name]]
        sampled.append(sum(values) / len(values))
    sampled.sort()
    return {"successes": sum(v for values in groups.values() for v in values), "n": n,
            "fraction": sum(v for values in groups.values() for v in values) / n,
            "bootstrap_95_percentile": [sampled[24], sampled[974]], "clusters": len(names),
            "warning": "four or fewer development templates; not scientific generalization evidence"}


def summarize(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    requests = load_lines(directory / "requests.jsonl")
    completed = [r for r in requests if r["event"] == "request_completed"]
    micro = load_lines(directory / "micro.jsonl")
    lookup = {s.scenario_id: s for s in scenarios()}
    for r in micro:
        if r["status"] == "completed":
            actual = parser_diagnostic(lookup[r["scenario_id"]], r["generation"]["text"])
            if actual != r["parser"]:
                raise ValueError("Parser diagnostic mismatch")
    episodes = [json.loads(p.read_text()) for p in sorted(directory.glob("episode_[0-9][0-9][0-9].json"))]
    for episode in episodes:
        verify_episode(episode)
    resources = load_lines(directory / "resources.jsonl")
    resident = [m for s in resources for m in s.get("resident_models", [])]
    summary = {"directory": str(directory), "stage": metadata["stage"], "status": metadata["status"],
               "model_key": metadata["model_key"], "model_digest": metadata["preflight"]["model"]["digest"],
               "quantization": metadata["preflight"]["precision"], "seed": 13,
               "ollama_version": metadata["preflight"]["ollama_version"],
               "scope": "exploratory_only_not_BF16_or_final_RQ_evidence",
               "source_snapshot_sha256": metadata["source_snapshot_sha256"],
               "requests_started": sum(r["event"] == "request_started" for r in requests),
               "requests_completed": len(completed),
               "request_failures": [r for r in requests if r["event"] == "request_failed"],
               "run_error": metadata.get("error"), "wall_seconds": metadata["wall_seconds"],
               "residents_after_unload": metadata.get("residents_after_unload"),
               "sampled_max_process_rss_bytes": max((s.get("sum_process_rss_bytes", 0) for s in resources), default=None),
               "sampled_max_resident_size_bytes": max((m.get("size", 0) for m in resident), default=None),
               "sampled_max_resident_size_vram_bytes": max((m.get("size_vram", 0) for m in resident), default=None),
               "memory_caveat": "sampled maxima, not guaranteed peaks; unified RSS/VRAM overlap, never add",
               "resource_samples": len(resources)}
    if completed:
        rates = [r["response"]["eval_count"] / (r["response"]["eval_duration"] / 1e9) for r in completed
                 if r["response"].get("eval_duration", 0) > 0]
        walls = [r["wall_seconds"] for r in completed]
        summary["inference"] = {"n": len(completed), "median_wall_seconds": statistics.median(walls),
                                "min_wall_seconds": min(walls), "max_wall_seconds": max(walls),
                                "median_decode_tokens_per_second": statistics.median(rates) if rates else None,
                                "load_seconds_first_request": completed[0]["response"].get("load_duration", 0) / 1e9,
                                "done_reasons": dict(Counter(r["response"].get("done_reason") for r in completed))}
    if metadata["stage"] == "cold" and completed:
        summary["cold_parser"] = parser_diagnostic(scenarios()[0], completed[0]["response"]["response"])
        if metadata["status"] != "completed":
            summary["unexecuted_due_to_cold_guard"] = {"planned_suite_micro_calls": 12, "planned_agent_episodes": 10}
            summary["post_request_swap_delta_bytes"] = None
            summary["missing_measurement"] = "first guard logged threshold failure but not its numeric reading"
    if micro:
        summary["micro"] = {}
        for condition in ("plain", "schema"):
            subset = [r for r in micro if r["condition"] == condition]
            summary["micro"][condition] = {field: rate(subset, field) for field in
                                            ("json_valid", "grammar_valid", "legal_action")}
        pairs = defaultdict(list)
        for r in micro:
            if r["condition"] == "plain":
                pairs[r["scenario_id"]].append(r)
        repeat = [{"id": key, "identical": len(values) == 2 and all(v["status"] == "completed" for v in values)
                   and values[0]["generation"]["text"] == values[1]["generation"]["text"]}
                  for key, values in pairs.items()]
        summary["micro_plain_exact_repeats"] = repeat
    if episodes:
        unique = [e for e in episodes if e["repetition"] == 1]
        all_turns = [t for e in episodes for t in e["trace"]]
        goal_rows = [{"template_id": e["template_id"], "parser": {"goal": e["goal_met_at_termination"]}} for e in unique]
        summary["episodes"] = {"assigned_including_repeats": len(episodes), "unique_assigned": len(unique),
                               "all_replays_verified": True, "unique_goal_completion": rate(goal_rows, "goal"),
                               "termination_counts": dict(Counter(e["status"] for e in episodes)),
                               "turns": len(all_turns), "rejected_turns": sum(t["status"] == "rejected" for t in all_turns),
                               "executed_world_tools": sum(t["status"] == "executed" for t in all_turns),
                               "all_final_worlds_unchanged": all(e["initial_hash"] == e["final_hash"] for e in episodes),
                               "repeat_trace_checks": []}
        for e in episodes:
            if e["repetition"] == 2:
                first = next(a for a in unique if a["scenario_id"] == e["scenario_id"])
                summary["episodes"]["repeat_trace_checks"].append({"id": e["scenario_id"],
                          "identical": trace_signature(first) == trace_signature(e)})
    summary["raw_file_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in directory.iterdir() if p.is_file()}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"scope": "exploratory_engineering_only", "bootstrap_seed": 9001, "bootstrap_resamples": 1000,
              "runs": [summarize(p.parent) for p in sorted(args.root.glob("*/metadata.json"))]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{k: r.get(k) for k in ("model_key", "stage", "status", "inference", "cold_parser", "run_error")}
                      for r in report["runs"]], indent=2))


if __name__ == "__main__":
    main()
