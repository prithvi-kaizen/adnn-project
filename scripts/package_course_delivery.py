"""Create a local tracked-file handoff with deterministic metadata and hashes."""

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    assert target.is_relative_to(ROOT / "output/course_fast_track") and not target.exists()
    for options in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        subprocess.run(["git", *options], cwd=ROOT, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    tracked = sorted(p for p in tracked if p and not p.startswith("output/course_fast_track/"))
    forbidden = (".git/", ".venv/", ".venv-research/", ".cache/", "docs/evidence/private/", "artifacts/activations/",
                 "output/pdf/annotation_handoff_pilot_v1/Give_to_Annotator_A/")
    forbidden_suffix = {".safetensors", ".gguf", ".bin", ".pem", ".key", ".pt", ".pth"}
    entries = []
    secret = re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b|\bghp_[A-Za-z0-9]{20,}\b|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----")
    text_suffix = {".py", ".sh", ".json", ".jsonl", ".md", ".txt", ".yaml", ".tex", ".bib", ".mjs"}
    for name in tracked:
        path = ROOT / name
        assert not name.startswith(forbidden) and path.suffix not in forbidden_suffix and path.name not in {".env", "token"}, name
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        raw = path.read_bytes()
        if path.suffix in text_suffix:
            assert not secret.search(raw), "Potential credential detected; do not package "+name
        entries.append({"path": name, "bytes": len(raw), "sha256": digest(raw)})
    manifest = {"scope": "private_local_reduced_course_handoff", "source_commit": commit,
                "deterministic_zip_timestamp": "2026-11-17T00:00:00; packaging convention, not experimental timestamp",
                "full_original_compliance": False, "files": entries,
                "excluded": ["untracked user annotation folder", "Git internals", "environments/caches/credentials/private inventories",
                             "pretrained weights", "large main activation and fitted-state payloads"],
                "raw_vectors_manifest": "results/course_fast_track/native_mac_cpu_v1/features_manifest.json"}
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in entries:
            item = zipfile.ZipInfo(entry["path"], date_time=(2026, 11, 17, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            archive.writestr(item, (ROOT / entry["path"]).read_bytes())
        item = zipfile.ZipInfo("HANDOFF_MANIFEST.json", date_time=(2026, 11, 17, 0, 0, 0))
        item.compress_type = zipfile.ZIP_DEFLATED
        item.external_attr = 0o100644 << 16
        archive.writestr(item, json.dumps(manifest, indent=2)+"\n")
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for entry in entries:
            raw = archive.read(entry["path"])
            assert len(raw) == entry["bytes"] and digest(raw) == entry["sha256"]
        assert len(archive.namelist()) == len(entries)+1
    receipt = {"status": "PASS_LOCAL_ARCHIVE_HASHES_AND_CRC", "archive": str(target.relative_to(ROOT)),
               "archive_bytes": target.stat().st_size, "archive_sha256": digest(target.read_bytes()),
               "source_commit": commit, "payload_files": len(entries), "payload_bytes": sum(r["bytes"] for r in entries),
               "manifest_sha256": digest((json.dumps(manifest, indent=2)+"\n").encode()),
               "external_submission": False, "full_original_compliance": False}
    target.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
