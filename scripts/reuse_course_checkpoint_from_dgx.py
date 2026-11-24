"""Reuse the exact verified project checkpoint over SSH, without another Hub pull."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from stage_course_checkpoint import ROOT, file_receipt, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve previous receipts")
    pinned = json.loads((ROOT / "docs/evidence/course_phase1_20261115/fallback_model_metadata.json").read_text())
    settings = json.loads((ROOT / "configs/course_native_pilot_v1.json").read_text())
    old = json.loads((ROOT / "artifacts/course_fast_track/native_cpu_pilot_v1/checkpoint_stage.json").read_text())
    assert old["status"] == "STAGED_VERIFIED" and old["revision"] == pinned["revision"]
    cache = ROOT / settings["native_cache"]
    repo = cache / "models--Qwen--Qwen2.5-0.5B-Instruct"
    snapshot, blobs = repo / "snapshots" / pinned["revision"], repo / "blobs"
    snapshot.mkdir(parents=True, exist_ok=True)
    blobs.mkdir(parents=True, exist_ok=True)
    report = {"status": "STARTED", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "model": pinned["repo"], "revision": pinned["revision"], "snapshot": str(snapshot),
              "acquisition": "verified_project_DGX_cache_via_resumable_SFTP", "source_Slurm_job": "36023",
              "upstream_model_download": False, "pretrained_model_loaded": False,
              "source_receipt_sha256": sha(ROOT / "artifacts/course_fast_track/native_cpu_pilot_v1/checkpoint_stage.json"),
              "source_script_sha256": sha(Path(__file__)), "files": {}, "copied": [], "deleted_own_incomplete_files": []}
    started = time.monotonic()
    try:
        missing = [f for f in pinned["files"] if not (snapshot / f["path"]).is_file()]
        reserve = sum(settings[k] for k in ("local_disk_reserve_bytes", "output_allowance_bytes", "temporary_allowance_bytes"))
        report.update(free_bytes_before=shutil.disk_usage(ROOT).free,
                      required_free_bytes=sum(f["bytes"] for f in missing)+reserve)
        assert report["free_bytes_before"] >= report["required_free_bytes"]
        report["capacity_passed"] = True
        for spec in pinned["files"]:
            link = snapshot / spec["path"]
            blob = blobs / spec["etag"]
            if link.is_file():
                report["files"][spec["path"]] = file_receipt(link, spec)
                continue
            temporary = blobs / (spec["etag"] + ".dgx-copy")
            remote = old["snapshot"] + "/" + spec["path"]
            resumed_bytes = temporary.stat().st_size if temporary.exists() else 0
            with tempfile.NamedTemporaryFile(mode="w", dir=blobs, prefix=".sftp-", delete=True) as batch:
                batch.write(f'reget "{remote}" "{temporary}"\n')
                batch.flush()
                command = ["sftp", "-o", "BatchMode=yes", "-o", "ControlPath=/tmp/adnn-course-%C",
                           "-b", batch.name, "se23uari141@10.59.121.172"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
            if result.returncode:
                raise RuntimeError("SFTP transfer failed; incomplete task-owned copy retained")
            receipt = file_receipt(temporary, spec)
            if blob.exists():
                file_receipt(blob, spec)
                temporary.unlink()
            else:
                temporary.rename(blob)
            if not link.exists():
                link.symlink_to(os.path.relpath(blob, snapshot))
            report["files"][spec["path"]] = file_receipt(link, spec)
            report["copied"].append({"path": spec["path"], "resumed_from_bytes": resumed_bytes, **receipt})
            # Only the exact interrupted download created by this task is disposable.
            incomplete = blobs / (spec["etag"] + ".incomplete")
            if incomplete.is_file() and incomplete.stat().st_uid == os.getuid():
                info = {"path": str(incomplete.relative_to(ROOT)), "bytes": incomplete.stat().st_size,
                        "allocated_bytes": incomplete.stat().st_blocks * 512}
                incomplete.unlink()
                report["deleted_own_incomplete_files"].append(info)
        report.update(status="STAGED_VERIFIED", free_bytes_after=shutil.disk_usage(ROOT).free)
        assert report["free_bytes_after"] >= reserve
    except Exception as error:
        report.update(status="FAILED", error_type=type(error).__name__, error=str(error))
    report["elapsed_seconds"] = time.monotonic()-started
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "acquisition", "elapsed_seconds")}, indent=2))
    if report["status"] != "STAGED_VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
