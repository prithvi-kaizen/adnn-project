"""Execute the preregistered excluded native pilot, or prepare its source lock."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.stage_course_checkpoint import file_receipt  # noqa: E402
from src.agent.course_metrics import audit_course_episode  # noqa: E402
from src.agent.loop import initial_messages, run_episode, verify_episode  # noqa: E402
from src.data.course import scenario_from_task  # noqa: E402
from src.data.splits import structural_state_signature  # noqa: E402
from src.models.course_native import (  # noqa: E402
    FEATURE_VERSION,
    STATED_SYSTEM,
    STATED_VERSION,
    NativeCourseRunner,
    append_record,
    parse_stated,
    stated_messages,
)
from src.models.native_smoke import steer_last_position  # noqa: E402
from src.models.ollama_dev import api, system_memory  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def pilot_exposure(output, task, data_dir, identity):
    rows = [json.loads(line) for line in (data_dir / "sandbox_pool.jsonl").read_text().splitlines()]
    aliases = {r["id"] for r in rows if structural_state_signature(r["state_text"]) ==
               structural_state_signature(task["state_text"])}
    overlaps = {}
    for seed in (1729, 2718, 31415):
        split = json.loads((data_dir / f"splits/pool_1800/seed_{seed}.json").read_text())
        if aliases & set(split["indices"]["test"]):
            raise ValueError("Excluded pilot aliases a course test component")
        overlaps[str(seed)] = sorted({split["components"][i] for i in aliases})
    append_record(output / "pilot_exposures.jsonl", {
        "scope": "previously_excluded_development_pilot", "id": task["id"],
        "state_hash": task["state_hash"], "model": identity, "timestamp_unix": time.time(),
        "before_any_pilot_model_output": True, "overlap_components_by_seed": overlaps,
        "course_test_overlap": False})


def hook_checks(runner, task, settings):
    scenario = scenario_from_task(task)
    messages = initial_messages(scenario)
    _, encoded = runner.encode(messages)
    ids = torch.tensor([encoded["input_ids"]], device=runner.device)
    args = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "use_cache": False}
    block = runner.model.model.layers[-1]
    hooks = len(block._forward_hooks)
    vector = torch.randn(runner.model.config.hidden_size, generator=torch.Generator().manual_seed(1729))
    vector /= vector.norm()
    vector = vector.to(runner.device)
    with torch.inference_mode():
        baseline = runner.model(**args).logits
        with steer_last_position(block, vector, 0):
            zero = runner.model(**args).logits
        assert torch.equal(baseline, zero), "Zero-strength logits differ"
        del zero
        with steer_last_position(block, vector, settings["hook_test_strength"]):
            perturbed = runner.model(**args).logits
        assert torch.equal(baseline[:, :-1], perturbed[:, :-1])
        assert not torch.equal(baseline[:, -1], perturbed[:, -1])
        del baseline, perturbed
    try:
        with steer_last_position(block, vector, 1):
            raise RuntimeError("intentional cleanup test")
    except RuntimeError:
        pass
    assert len(block._forward_hooks) == hooks
    ordinary = runner.generate(messages)
    with steer_last_position(block, vector, 0):
        zero = runner.generate(messages)
    repeated = runner.generate(messages)
    assert ordinary["output_token_ids"] == zero["output_token_ids"] == repeated["output_token_ids"]
    shapes = []
    observer = block.register_forward_hook(lambda module, inputs, output: shapes.append(list(output[0].shape)))
    try:
        with steer_last_position(block, vector, settings["hook_test_strength"]):
            runner.generate(messages, max_new_tokens=4, min_new_tokens=3)
    finally:
        observer.remove()
    assert len(block._forward_hooks) == hooks
    # An immediate EOS can yield no decode step; preserve failure rather than assuming one.
    assert any(shape[1] == 1 for shape in shapes), "No cached one-token hook call observed"
    return {"zero_strength_logits_identical": True, "zero_strength_complete_greedy_ids_identical": True,
            "repeated_greedy_ids_identical": True, "nonzero_only_changes_last_position_logits": True,
            "hook_cleanup_on_exception": True, "cached_hook_shapes": shapes,
            "ordinary_request_id": ordinary["request_id"], "zero_request_id": zero["request_id"],
            "repeat_request_id": repeated["request_id"], "direction_is_train_reversibility_direction": False,
            "is_bonus_RQ4_result": False}


def select_pool(settings, width, blocks, feature_max_seconds, stated_max_seconds):
    rules = settings["pool_selection"]
    projections = {}
    selected = None
    for name, n in (("pool_1800", 1800), ("subset_360", 360)):
        seconds = rules["runtime_multiplier"] * n * (feature_max_seconds + stated_max_seconds)
        feature_bytes = n * width * 4 * (2 * (blocks + 1) + 2)
        artifact_bytes = feature_bytes + rules["reserve_for_logs_and_results_bytes"]
        fits = seconds <= rules["extraction_plus_stated_budget_seconds"] and artifact_bytes <= rules["artifact_allowance_bytes"]
        projections[name] = {"N": n, "projected_extraction_plus_stated_seconds": seconds,
                             "raw_FP32_feature_bytes": feature_bytes,
                             "projected_bytes_with_log_reserve": artifact_bytes, "fits": fits}
        if selected is None and fits:
            selected = name
    if selected is None:
        raise RuntimeError("Neither predeclared inference pool fits the measured projection")
    return {"selected_pool": selected, "projections": projections,
            "uses_model_accuracy_or_agent_success": False, "course_test_outcomes_seen": 0,
            "selected_at_utc": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--checkpoint-receipt", type=Path)
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/course_native_pilot_v1.json")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Preserve previous attempts; choose a new output directory")
    args.output_dir.mkdir(parents=True)
    out = args.output_dir
    settings_path = args.settings.resolve()
    settings = json.loads(settings_path.read_text())
    transported = ROOT / "source_provenance.json"
    if transported.is_file():
        provenance = json.loads(transported.read_text())
        assert all(sha(ROOT / name) == value for name, value in provenance["files_sha256"].items())
        source_commit = provenance["source_commit"]
        source_dirty = provenance["worktree_dirty"]
    else:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        source_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    data_dir = ROOT / "data/course_fast_track/v2"
    tasks = [json.loads(line) for line in (data_dir / "development_tasks.jsonl").read_text().splitlines()]
    paths = [Path(__file__).resolve(), settings_path, ROOT / "src/models/course_native.py",
             ROOT / "src/agent/course_metrics.py", ROOT / "src/agent/loop.py",
             ROOT / "src/models/native_smoke.py", ROOT / "src/models/alignment.py",
             ROOT / ("requirements-runtime-dgx-lock.txt" if settings.get("require_Slurm_CPU_allocation") else
                     "requirements-runtime-mac-lock.txt"), ROOT / "docs/COURSE_NATIVE_PILOT_PROTOCOL.md"]
    if settings.get("loader_amendment"):
        paths.append(ROOT / settings["loader_amendment"])
    if settings.get("resource_amendment"):
        paths.append(ROOT / settings["resource_amendment"])
    report = {"scope": settings["scope"], "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_commit": source_commit, "worktree_dirty": source_dirty,
              "transported_source_manifest": transported.is_file(),
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths}, "settings": settings,
              "course_freeze_sha256": sha(data_dir / "freeze_manifest.json"),
              "feature_prompt_version": FEATURE_VERSION, "stated_prompt_version": STATED_VERSION,
              "stated_system_sha256": hashlib.sha256(STATED_SYSTEM.encode()).hexdigest(),
              "pretrained_model_loaded": False, "weights_downloaded_by_pilot": False,
              "course_holdout_model_outputs": 0, "python": sys.version, "platform": platform.platform(),
              "packages": {p: importlib.metadata.version(p) for p in (
                  "torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "numpy", "scikit-learn")}}
    save(out / "source_lock.json", report)
    if not args.run:
        report["status"] = "PREPARED_NOT_EXECUTED"
        save(out / "report.json", report)
        print("Prepared native-pilot source lock; no model loaded or downloaded.")
        return
    started = time.monotonic()
    try:
        frozen = json.loads((data_dir / "freeze_manifest.json").read_text())
        assert all(sha(data_dir / p) == entry["sha256"] for p, entry in frozen["files"].items())
        if settings.get("require_Slurm_CPU_allocation"):
            assert os.environ.get("SLURM_JOB_ID") and not platform.node().startswith("dgx-login")
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            assert int(os.environ["SLURM_CPUS_PER_TASK"]) >= settings["CPU_threads"]
            report.update(slurm_job_id=os.environ["SLURM_JOB_ID"], node=platform.node(), GPU_experiment=False)
        if args.checkpoint_receipt is None:
            raise ValueError("An actual verified checkpoint receipt is required")
        checkpoint = json.loads(args.checkpoint_receipt.read_text())
        assert checkpoint["status"] == "STAGED_VERIFIED"
        assert (checkpoint["model"], checkpoint["revision"]) == (settings["model"], settings["revision"])
        snapshot = ROOT / checkpoint["snapshot"]
        pinned = json.loads((ROOT / "docs/evidence/course_phase1_20261115/fallback_model_metadata.json").read_text())
        report["checkpoint_receipt_sha256"] = sha(args.checkpoint_receipt)
        report["checkpoint_files"] = {f["path"]: file_receipt(snapshot / f["path"], f) for f in pinned["files"]}
        def read_memory():
            if sys.platform == "darwin":
                return system_memory()
            memory = {line.split(":", 1)[0]: int(line.split()[1]) for line in
                      Path("/proc/meminfo").read_text().splitlines() if len(line.split()) >= 2}
            swap = next(line for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmSwap:"))
            return {"memory_free_percentage": int(100 * memory["MemAvailable"] / memory["MemTotal"]),
                    "swap_used_bytes": int(swap.split()[1]) * 1024,
                    "swap_measurement_scope": "this_Slurm_Python_process_not_other_users"}
        if sys.platform == "darwin" and api("/api/ps", timeout=5).get("models", []):
            raise RuntimeError("Another Ollama model is resident; do not evict unrelated work")
        baseline = read_memory()
        report["system_memory_baseline"] = baseline
        if settings.get("prior_guard_failure_report"):
            prior_path = ROOT / settings["prior_guard_failure_report"]
            prior = json.loads(prior_path.read_text())
            assert prior["status"] == "FAILED_OR_BLOCKED" and prior["course_holdout_model_outputs"] == 0
            report["prior_failed_loading_condition"] = {
                "report": settings["prior_guard_failure_report"], "sha256": sha(prior_path),
                "baseline": prior["system_memory_baseline"], "error": prior["error"],
                "fresh_baseline_for_changed_loader_not_within_old_attempt": True}
        if sys.platform == "darwin":
            report["unified_memory_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
        report["execution_platform"] = settings.get("execution_platform", "Mac_MPS")
        report["GPU_experiment"] = settings["device"] == "mps"

        def guard():
            measured = read_memory()
            measured.update(timestamp_unix=time.time(), free_disk_bytes=shutil.disk_usage(ROOT).free)
            measured["swap_growth_bytes"] = measured["swap_used_bytes"] - baseline["swap_used_bytes"]
            append_record(out / "resource_guards.jsonl", measured)
            if (measured["swap_growth_bytes"] > settings["guard_swap_growth_bytes"] or
                    measured["memory_free_percentage"] < settings["guard_minimum_memory_free_percentage"] or
                    measured["free_disk_bytes"] < settings["local_disk_reserve_bytes"] or
                    time.monotonic() - started > settings["pilot_runtime_limit_seconds"]):
                raise RuntimeError("Resource guard failed; preserve this attempt without rebaselining")

        random.seed(settings["seed"])
        np.random.seed(settings["seed"])
        torch.manual_seed(settings["seed"])
        torch.set_num_threads(settings["CPU_threads"])
        torch.use_deterministic_algorithms(True, warn_only=True)
        identity = {"model": settings["model"], "revision": settings["revision"],
                    "precision": settings["precision"], "device": settings["device"]}
        for task in tasks:
            pilot_exposure(out, task, data_dir, identity)
        with warnings.catch_warnings(record=True) as observed_warnings:
            warnings.simplefilter("always")
            runner = NativeCourseRunner(snapshot, settings, out, guard)
            report.update(pretrained_model_loaded=True, load_seconds=runner.load_seconds,
                          parameter_count=sum(p.numel() for p in runner.model.parameters()),
                          architecture=type(runner.model).__name__,
                          hidden_size=runner.model.config.hidden_size,
                          transformer_blocks=runner.model.config.num_hidden_layers,
                          low_cpu_mem_usage=settings.get("low_cpu_mem_usage", True),
                          load_directly_to_device=settings.get("load_directly_to_device", False),
                          parameter_devices=sorted({str(p.device) for p in runner.model.parameters()}),
                          chat_template_sha256=hashlib.sha256(runner.tokenizer.chat_template.encode()).hexdigest(),
                          tokenizer_class=type(runner.tokenizer).__name__, padding_side=runner.tokenizer.padding_side,
                          CPU_fallback_enabled=False, deterministic_algorithms_warn_only=True)
            save(out / "loaded_checkpoint.json", report)
            assert all(p.device.type == settings["device"] for p in runner.model.parameters())
            batches = []
            for task in tasks:
                result = runner.feature_batch([task])
                for record in result["records"]:
                    append_record(out / "feature_prompts.jsonl", record)
                batches.append(result)
            padded = runner.feature_batch([tasks[0], tasks[2]])
            for i, index in enumerate((0, 2)):
                for pooling in ("last", "mean"):
                    assert torch.allclose(padded[pooling][:, i], batches[index][pooling][:, 0],
                        atol=settings["padding_atol"], rtol=settings["padding_rtol"]), "Padding changes pooled features"
                assert padded["records"][i]["selected_action_token_ids"] == batches[index]["records"][0]["selected_action_token_ids"]
            arrays = {key: torch.cat([batch[key] for batch in batches], dim=(1 if key in ("last", "mean") else 0)).numpy()
                      for key in ("last", "mean", "raw_final_last", "raw_final_mean")}
            assert all(np.isfinite(value).all() for value in arrays.values())
            with torch.inference_mode():
                normalized = runner.model.model.norm(torch.from_numpy(arrays["raw_final_last"]).to(runner.device)).cpu().numpy()
            assert np.allclose(normalized, arrays["last"][-1], atol=1e-4, rtol=1e-4)
            assert not np.array_equal(arrays["raw_final_last"], arrays["last"][-1])
            np.savez_compressed(out / "features.npz", **arrays)
            report["features"] = {"ids": [t["id"] for t in tasks],
                "shapes": {k: list(v.shape) for k, v in arrays.items()},
                "file_sha256": sha(out / "features.npz"), "file_bytes": (out / "features.npz").stat().st_size,
                "single_case_seconds": [b["seconds"] for b in batches],
                "memory_snapshots": [b["memory"] for b in batches],
                "padding_check": "PASS", "raw_final_is_distinct": True}
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            train = [i for i, t in enumerate(tasks) if "-2-" not in t["id"]]
            sanity = [i for i, t in enumerate(tasks) if "-2-" in t["id"]]
            x, y = arrays["last"][12], np.array([t["label"] for t in tasks])
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000,
                                  class_weight="balanced", random_state=1729))
            probe.fit(x[train], y[train])
            probabilities = probe.predict_proba(x[sanity])
            assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1)
            save(out / "linear_slice.json", {"scope": "shared_template_excluded_engineering_only",
                "training_ids": [tasks[i]["id"] for i in train], "sanity_ids": [tasks[i]["id"] for i in sanity],
                "predictions": probe.predict(x[sanity]).tolist(), "probabilities": probabilities.tolist(),
                "classes": probe.classes_.tolist(), "scientific_accuracy_reported": False, "fixed_layer": 12})
            stated = []
            for task in tasks:
                response = runner.generate(stated_messages(task), max_new_tokens=settings["stated_max_new_tokens"])
                record = {"id": task["id"], "parsed_label": parse_stated(response["text"]), **response}
                append_record(out / "stated.jsonl", record)
                stated.append(record)
            report["stated"] = {"N": len(stated), "valid_JSON_labels": sum(r["parsed_label"] is not None for r in stated),
                                 "seconds": [r["seconds"] for r in stated]}
            report["hook_checks"] = hook_checks(runner, tasks[0], settings)
            episodes = []
            for task_id in settings["agent_pilot_ids"]:
                task = next(t for t in tasks if t["id"] == task_id)
                episode = run_episode(scenario_from_task(task), runner.generate,
                    record_event=lambda event: append_record(out / "agent_events.jsonl", {"id": task_id, **event}))
                verify_episode(episode)
                save(out / f"episode_{task_id}.json", episode)
                audit = audit_course_episode(task, episode)
                save(out / f"audit_{task_id}.json", audit)
                if audit["infrastructure_failure"]:
                    raise RuntimeError("Pilot agent has an infrastructure/context failure")
                episodes.append({"id": task_id, "status": episode["status"], "steps": episode["steps"],
                                 "goal_success": audit["goal_success"], "caution": audit["caution"],
                                 "rejected_calls": audit["rejected_calls"], "replay_verified": True})
            report["agent_episodes"] = episodes
            decision = select_pool(settings, runner.model.config.hidden_size, runner.model.config.num_hidden_layers,
                                   max(b["seconds"] for b in batches), max(r["seconds"] for r in stated))
            save(out / "pool_decision.json", decision)
            report["pool_decision"] = decision
            generation_rows = [json.loads(line) for line in (out / "generation_responses.jsonl").read_text().splitlines()]
            worst_seconds_per_token = max(row["seconds"] / max(1, row["new_tokens"]) for row in generation_rows)
            report["behavior_runtime_projection"] = {
                "assigned_versions": 24, "max_turns": 15, "max_tokens_per_turn": 128,
                "worst_observed_seconds_per_token_including_prefill": worst_seconds_per_token,
                "twofold_full_budget_seconds": 2 * 24 * 15 * 128 * worst_seconds_per_token,
                "does_not_assume_early_parse_failure": True,
                "longer_context_can_exceed_projection": True,
                "RQ4_17_condition_sweep_launched": False}
            report["memory_at_end"] = runner.memory()
            report["warnings"] = sorted({str(w.message) for w in observed_warnings})
            report["status"] = "PASS_C3_REDUCED_NATIVE_ENGINEERING_SCOPE"
            del runner
            if settings["device"] == "mps":
                torch.mps.empty_cache()
    except Exception as error:
        report.update(status="FAILED_OR_BLOCKED", error_type=type(error).__name__, error=str(error))
    report["elapsed_seconds"] = time.monotonic()-started
    save(out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "pretrained_model_loaded", "course_holdout_model_outputs")}, indent=2))
    if report["status"] != "PASS_C3_REDUCED_NATIVE_ENGINEERING_SCOPE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
