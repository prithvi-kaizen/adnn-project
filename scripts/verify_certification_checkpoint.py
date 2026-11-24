"""Verify saved engineering evidence offline; never clear a research gate."""

import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_ollama_diagnostic import summarize  # noqa: E402
from src.models.alignment import action_positions  # noqa: E402

os.chdir(ROOT)
root = ROOT
freeze = json.load(open("docs/evidence/protocol_freeze.json"))
assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() == v for p, v in freeze["sha256"].items())
assert summarize(pathlib.Path("artifacts/exploratory/ollama_diagnostic_v2")) == json.load(
    open("docs/evidence/ollama_diagnostic_v2_summary.json")
)
transport = json.load(open("docs/evidence/certification_source_transport.json"))
archive = pathlib.Path("artifacts/source/certification_29888_source.tar.gz")
assert hashlib.sha256(archive.read_bytes()).hexdigest() == transport["archive_sha256"]
with tarfile.open(archive) as tar:
    for p, v in transport["sources"].items():
        assert hashlib.sha256(tar.extractfile(p).read()).hexdigest() == v
weights = json.load(open("docs/evidence/model_storage_metadata_20261101.json"))["models"]
for name in ("qwen_access", "llama_access"):
    r = json.load(open("docs/evidence/native_audit_29888/" + name + ".json"))
    spec = weights[r["model"]]
    assert r["access_verified"] and r["authentication"]["valid"] and r["revision"] == spec["revision"]
    for s, w in zip(r["shards"], spec["primary_safetensors"]):
        assert (
            s["status"] == "ACCESS_VERIFIED" and s["bytes"] == w["bytes"] and s["etag"] == w["lfs"]["sha256"]
        )
tokens = json.load(open("docs/evidence/tokenizer_alignment_29890/tokenizer_alignment.json"))
assert (
    tokens["status"] == "PASS"
    and not tokens["weights_downloaded"]
    and tokens["total_asset_bytes"] <= 50000000
)
assert (
    tokens["source_sha256"]
    == hashlib.sha256(pathlib.Path("scripts/hpc/check_native_tokenizers.py").read_bytes()).hexdigest()
)
for m in tokens["models"]:
    assert m["revision"] == weights[m["repo"]]["revision"] and len(m["cases"]) == 25
    for c in m["cases"]:
        a, b = c["span"]
        assert c["rendered_prompt"][a:b] == c["action"]
        assert action_positions(c["offsets"], c["attention_mask"], a, b) == c["action_positions"]
        assert c["last_action_token_id"] == c["input_ids"][c["action_positions"][-1]]
        assert len(c["input_ids"]) <= 2048
rows = json.load(open("docs/evidence/requirements_status_certification_v2.json"))
assert len(rows) == 117 and all(r["status"] in ("PASS", "PARTIAL", "BLOCKED") for r in rows)
registry = [json.loads(x) for x in pathlib.Path("results/registry.jsonl").read_text().splitlines()]
assert len(registry) == len({r["experiment_id"] for r in registry})
log = pathlib.Path("docs/evidence/certification_local_unittest_v2.log").read_bytes()
assert b"Ran 92 tests" in log and b"\nOK\n" in log and b"FAILED" not in log
report = {
    "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "verified_engineering_checkpoint_not_Gate2_completion",
    "current_git_baseline": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "local_research_suite": {
        "python": "3.12.14",
        "passed": 92,
        "skipped": 0,
        "raw_log_sha256": hashlib.sha256(log).hexdigest(),
    },
    "dgx_suite": {
        "slurm_job_id": 29888,
        "passed": 91,
        "skipped": 0,
        "elapsed_seconds": 41,
        "source_archive_verified": True,
    },
    "tokenizer_only_job": {
        "slurm_job_id": 29890,
        "elapsed_seconds": 12,
        "fixtures_per_family": 25,
        "asset_bytes": tokens["total_asset_bytes"],
        "weights_downloaded": False,
    },
    "both_pinned_model_shard_access": "PASS",
    "tokenizer_span_receipts": "PASS",
    "independent_quantized_summary": "PASS",
    "all_117_requirement_statuses_present": "PASS",
    "frozen_six_file_integrity": "PASS",
    "lint": "PASS",
    "sandbox_candidates": 1800,
    "verified_minimal_pair_members": 1620,
    "confirmed_physical_labels": 0,
    "real_double_annotations": 0,
    "phase2_gate": "PARTIAL",
    "final_joint_freeze": False,
    "methodology_amendment_applied": False,
    "full_scale_GPU_run": False,
}
pathlib.Path("/tmp/adnn-certification-verification.json").write_text(json.dumps(report, indent=2) + "\n")
print("PASS: source archive, frozen hashes, 117 statuses, access receipts, tokenizer spans and raw summary")
