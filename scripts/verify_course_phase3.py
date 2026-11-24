"""Recheck saved native pilot evidence without loading the model or networking."""

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_course_native_pilot import select_pool  # noqa: E402
from src.agent.course_metrics import audit_course_episode  # noqa: E402
from src.agent.loop import verify_episode  # noqa: E402
from src.models.alignment import action_positions  # noqa: E402
from src.models.course_native import parse_stated  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(run_dir):
    r = read(run_dir / "report.json")
    assert r["status"] == "PASS_C3_REDUCED_NATIVE_ENGINEERING_SCOPE"
    assert r["pretrained_model_loaded"] and r["course_holdout_model_outputs"] == 0
    device = r["settings"]["device"]
    is_dgx = bool(r.get("slurm_job_id"))
    assert device in {"cpu", "mps"}
    if device == "cpu" and is_dgx:
        assert not r["GPU_experiment"] and r["slurm_job_id"] == "36023"
    assert r["settings"]["precision"] == "float32" and not r["CPU_fallback_enabled"]
    assert (r["parameter_count"], r["transformer_blocks"], r["hidden_size"]) == (494032768, 24, 896)
    source_name = ("source_transport.json" if is_dgx else
                   "source_snapshot_mac_cpu_complete.json" if device == "cpu" else
                   "source_snapshot_mps_direct.json" if r["settings"].get("load_directly_to_device") else
                   "source_snapshot_mps.json")
    transport_path = ROOT / "docs/evidence/course_phase3_20261115" / source_name
    transport = read(transport_path)
    archive_path = ROOT / transport["archive"]
    assert sha(archive_path) == transport["archive_sha256"]
    with tarfile.open(archive_path) as archive:
        for name, expected in transport["files"].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected, name
        for name, expected in r["source_sha256"].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected, name
    original = read(ROOT / "docs/evidence/protocol_freeze.json")
    assert all(sha(ROOT / name) == expected for name, expected in original["sha256"].items())
    assert sha(ROOT / "data/course_fast_track/v2/freeze_manifest.json") == r["course_freeze_sha256"]
    checkpoint = read(run_dir.parent / "checkpoint_stage.json")
    assert checkpoint["status"] == "STAGED_VERIFIED" and checkpoint["capacity_passed"]
    pinned = read(ROOT / "docs/evidence/course_phase1_20261115/fallback_model_metadata.json")
    assert checkpoint["revision"] == r["settings"]["revision"] == pinned["revision"]
    assert sum(f["bytes"] for f in checkpoint["files"].values()) == 995137675
    assert checkpoint["files"]["model.safetensors"]["sha256"] == next(
        f["lfs_sha256"] for f in pinned["files"] if f["path"] == "model.safetensors")
    features_path = run_dir / "features.npz"
    assert sha(features_path) == r["features"]["file_sha256"]
    with np.load(features_path, allow_pickle=False) as features:
        for name, shape in r["features"]["shapes"].items():
            assert list(features[name].shape) == shape and np.isfinite(features[name]).all()
            assert features[name].dtype == np.float32
        assert features["last"].shape == features["mean"].shape == (25, 24, 896)
        assert features["raw_final_last"].shape == features["raw_final_mean"].shape == (24, 896)
        assert not np.array_equal(features["raw_final_last"], features["last"][-1])
    prompts = rows(run_dir / "feature_prompts.jsonl")
    assert [p["id"] for p in prompts] == r["features"]["ids"] and len(prompts) == 24
    for prompt in prompts:
        positions = action_positions(prompt["offset_mapping"], prompt["attention_mask"], *prompt["action_span"])
        assert positions == prompt["action_positions"]
        assert [prompt["input_ids"][i] for i in positions] == prompt["selected_action_token_ids"]
    stated = rows(run_dir / "stated.jsonl")
    assert [s["id"] for s in stated] == r["features"]["ids"]
    assert all(parse_stated(s["text"]) == s["parsed_label"] for s in stated)
    assert sum(s["parsed_label"] is not None for s in stated) == r["stated"]["valid_JSON_labels"]
    responses = {row["request_id"]: row for row in rows(run_dir / "generation_responses.jsonl")}
    hooks = r["hook_checks"]
    assert all(hooks[key] for key in ("zero_strength_logits_identical", "zero_strength_complete_greedy_ids_identical",
                                     "repeated_greedy_ids_identical", "nonzero_only_changes_last_position_logits",
                                     "hook_cleanup_on_exception"))
    assert any(s[1] == 1 for s in hooks["cached_hook_shapes"])
    assert (responses[hooks["ordinary_request_id"]]["output_token_ids"] ==
            responses[hooks["zero_request_id"]]["output_token_ids"] ==
            responses[hooks["repeat_request_id"]]["output_token_ids"])
    assert not hooks["is_bonus_RQ4_result"]
    tasks = {row["id"]: row for row in rows(ROOT / "data/course_fast_track/v2/development_tasks.jsonl")}
    linear = read(run_dir / "linear_slice.json")
    assert len(linear["training_ids"]) == 16 and len(linear["sanity_ids"]) == 8
    assert set(linear["training_ids"]).isdisjoint(linear["sanity_ids"])
    assert set(linear["training_ids"] + linear["sanity_ids"]) == set(r["features"]["ids"])
    assert not linear["scientific_accuracy_reported"]
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    with np.load(features_path, allow_pickle=False) as features:
        index = {name: i for i, name in enumerate(r["features"]["ids"])}
        train = [index[name] for name in linear["training_ids"]]
        sanity = [index[name] for name in linear["sanity_ids"]]
        labels = np.array([tasks[name]["label"] for name in r["features"]["ids"]])
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000,
                              class_weight="balanced", random_state=1729))
        x = features["last"][linear["fixed_layer"]]
        probe.fit(x[train], labels[train])
        assert probe.predict(x[sanity]).tolist() == linear["predictions"]
        saved_probabilities = np.asarray(linear["probabilities"])
        assert saved_probabilities.shape == (8, 3) and np.isfinite(saved_probabilities).all()
        assert np.allclose(saved_probabilities.sum(1), 1)
        refit_probability_delta = float(np.max(np.abs(probe.predict_proba(x[sanity])-saved_probabilities)))
    for summary in r["agent_episodes"]:
        episode = read(run_dir / f"episode_{summary['id']}.json")
        verify_episode(episode)
        assert audit_course_episode(tasks[summary["id"]], episode) == read(run_dir / f"audit_{summary['id']}.json")
    decision = select_pool(r["settings"], 896, 24, max(r["features"]["single_case_seconds"]), max(r["stated"]["seconds"]))
    saved = read(run_dir / "pool_decision.json")
    assert decision["selected_pool"] == saved["selected_pool"] and decision["projections"] == saved["projections"]
    exposure = rows(run_dir / "pilot_exposures.jsonl")
    assert len(exposure) == 24 and all(e["before_any_pilot_model_output"] and not e["course_test_overlap"] for e in exposure)
    guards = rows(run_dir / "resource_guards.jsonl")
    assert max(e["timestamp_unix"] for e in exposure) < min(g["timestamp_unix"] for g in guards)
    if is_dgx:
        for p, count in ((run_dir.parent / "native_tests.log", 12), (run_dir.parent / "alignment_tests.log", 3)):
            log = p.read_text()
            assert f"Ran {count} tests" in log and "\nOK\n" in log
    else:
        tests = read(ROOT / "docs/evidence/course_phase3_20261115/unittest_receipt.json")
        assert tests["status"] == "PASS" and tests["tests"] >= 123
        assert all(sha(ROOT / p) == expected for p, expected in tests["source_sha256"].items())
    condition = "DGX_CPU" if is_dgx else "MAC_CPU" if device == "cpu" else "MPS"
    return {"course_gate_C3": f"PASS_REDUCED_NATIVE_{condition}_SCOPE", "original_research_gates": "PARTIAL",
            "actual_model": r["settings"]["model"], "actual_revision": r["settings"]["revision"],
            "actual_device": "DGX_allocated_CPU" if is_dgx else "Mac_CPU" if device == "cpu" else "Mac_MPS",
            "actual_precision": "float32", "slurm_job_id": r.get("slurm_job_id"),
            "excluded_feature_cases": 24, "stated_valid_JSON": r["stated"]["valid_JSON_labels"],
            "agent_pilot_episodes": len(r["agent_episodes"]), "selected_pool": saved["selected_pool"],
            "refit_label_predictions_identical": True,
            "refit_probability_max_abs_difference": refit_probability_delta,
            "refit_probability_equivalence_claimed": False,
            "course_holdout_model_outcomes": 0, "human_judgments": 0, "RQ4_result": False,
            "report_sha256": sha(run_dir / "report.json"), "features_sha256": sha(features_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new verification receipt")
    result = verify(args.run_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
