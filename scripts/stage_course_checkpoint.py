"""Budget, pin, stage once through hf, and verify every checkpoint file."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path, spec):
    if path.stat().st_size != spec["bytes"]:
        raise ValueError("Cached file size differs from pinned metadata")
    actual = sha(path)
    if spec["lfs_sha256"]:
        if actual != spec["lfs_sha256"]:
            raise ValueError("Safetensors SHA256 differs from pinned LFS identity")
    else:
        content = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != spec["etag"]:
            raise ValueError("Small asset differs from pinned Git blob identity")
    return {"bytes": path.stat().st_size, "sha256": actual, "pinned_identity_verified": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/course_native_pilot_v1.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve prior receipts; choose a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    settings = json.loads(args.settings.read_text())
    pinned = json.loads((ROOT / "docs/evidence/course_phase1_20261115/fallback_model_metadata.json").read_text())
    assert (settings["model"], settings["revision"]) == (pinned["repo"], pinned["revision"])
    cache = ROOT / Path(settings["native_cache"]).expanduser()
    snapshot = cache / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / settings["revision"]
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "model": settings["model"], "revision": settings["revision"],
              "source_sha256": sha(Path(__file__)), "settings_sha256": sha(args.settings),
              "snapshot": str(snapshot), "download_requested": args.download,
              "weight_download_attempted": False,
              "weights_downloaded_in_this_attempt": False, "pretrained_model_loaded": False}
    try:
        present = {f["path"]: file_receipt(snapshot / f["path"], f) for f in pinned["files"]
                   if (snapshot / f["path"]).is_file()}
        missing = [f for f in pinned["files"] if f["path"] not in present]
        reserve = sum(settings[k] for k in (
            "local_disk_reserve_bytes", "output_allowance_bytes", "temporary_allowance_bytes"))
        required = sum(f["bytes"] for f in missing) + reserve
        free = shutil.disk_usage(ROOT).free
        if settings.get("binding_home_quota_bytes"):
            if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_CPUS_PER_TASK"):
                raise ValueError("DGX staging/quota scan requires an allocated compute job")
            home_bytes = int(subprocess.check_output(["du", "-sx", "-B1", str(Path.home())], text=True).split()[0])
            quota_margin = settings["binding_home_quota_bytes"] - home_bytes
            report.update(home_allocated_bytes=home_bytes, binding_home_quota_bytes=settings["binding_home_quota_bytes"],
                          quota_margin_bytes=quota_margin)
            free = min(free, quota_margin)
        report.update(free_bytes=free, required_free_bytes=required,
                      missing_checkpoint_bytes=sum(f["bytes"] for f in missing),
                      shortfall_bytes=max(0, required-free), present=present,
                      capacity_passed=free >= required)
        if free < required:
            report["status"] = "BLOCKED_STORAGE"
        elif not args.download:
            report["status"] = "READY_TO_STAGE" if missing else "STAGED_VERIFIED"
        else:
            # Metadata requests do not fetch weights or use an account credential.
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
            for spec in pinned["files"]:
                meta = get_hf_file_metadata(hf_hub_url(settings["model"], spec["path"],
                    revision=settings["revision"]), token=False, timeout=15)
                assert (meta.commit_hash, meta.size, meta.etag) == (
                    settings["revision"], spec["bytes"], spec["etag"])
            if missing:
                env = dict(os.environ)
                env.update(HF_HUB_DISABLE_XET="1", HF_XET_CHUNK_CACHE_SIZE_BYTES="0",
                           HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
                env.pop("HF_TOKEN", None)
                command = [str(Path(sys.executable).parent / "hf"), "download", settings["model"],
                           *[f["path"] for f in missing], "--revision", settings["revision"],
                           "--cache-dir", str(cache), "--max-workers", "1"]
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
                safe_log = re.sub(r"https?://[^\s]+", "<URL omitted>", result.stdout + result.stderr)
                safe_log = re.sub(r"hf_[A-Za-z0-9]{20,}", "<credential omitted>", safe_log)
                args.output.with_suffix(".download.log").write_text(safe_log)
                report.update(download_exit_code=result.returncode,
                              weight_download_attempted=any(f["path"] == "model.safetensors" for f in missing))
                if result.returncode:
                    raise RuntimeError("hf download failed; sanitized log preserved")
            report["files"] = {f["path"]: file_receipt(snapshot / f["path"], f) for f in pinned["files"]}
            report["weights_downloaded_in_this_attempt"] = report["weight_download_attempted"]
            report["free_bytes_after"] = shutil.disk_usage(ROOT).free
            assert report["free_bytes_after"] >= reserve
            report["status"] = "STAGED_VERIFIED"
    except Exception as error:
        report.update(status="FAILED", error_type=type(error).__name__)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "model", "download_requested",
        "weights_downloaded_in_this_attempt")}, indent=2))
    if report["status"] in {"FAILED", "BLOCKED_STORAGE"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
