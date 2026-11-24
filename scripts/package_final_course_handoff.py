"""Merge the verified course releases into one locally auditable archive."""

import argparse
import hashlib
import json
import shutil
import zipfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "output/course_fast_track"
OUTPUT = PACKAGE_DIR / "adnn_course_final_v1.zip"
MANIFEST = "FINAL_HANDOFF_MANIFEST.json"
OLD_RELEASES = ("course_handoff_v4", "course_handoff_mac_v1", "course_handoff_mac_v2", "course_codex_review_v1")
OLD_ZIPS = {f"output/course_fast_track/{name}.zip" for name in OLD_RELEASES}
OLD_MANIFESTS = {"HANDOFF_MANIFEST.json", "MAC_HANDOFF_MANIFEST.json", "LLAMA_HANDOFF_MANIFEST.json",
                 "CODEX_REVIEW_MANIFEST.json"}
FINAL_FILES = (
    "README.md", "docs/COURSE_SUBMISSION_GUIDE.md", "docs/PHASE_STATUS.md", "docs/REQUIREMENTS_MATRIX.md",
    "docs/RISK_REGISTER.md", "docs/CONTRIBUTIONS.md", "docs/COURSE_FULL_EPISODE_STEERING_DIAGNOSTIC.md",
    "docs/COURSE_FULL_EPISODE_STEERING_RESULTS.md",
    "docs/COURSE_SCOPE_CONFIRMATION_20261120.md", "docs/SCIENCEWORLD_AMENDMENT_FOR_REVIEW.md",
    "docs/evidence/course_release_20261120", "scripts/run_course_full_episode_diagnostic.py",
    "scripts/run_mac_guarded_process.py", "scripts/build_course_acl_talk.mjs", "scripts/build_mac_acl_paper.py",
    "scripts/package_final_course_handoff.py", "artifacts/course_fast_track/native_full_episode_steering_v1",
    "artifacts/course_fast_track/native_full_episode_steering_guard_v1", "output/pdf/course_acl_mac_v4",
    "output/presentation/course_acl_final", "results/registry.jsonl",
)


def read_json(path):
    return json.loads(path.read_text())


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name or "\\" in name:
        raise ValueError(f"Unsafe archive path {name!r}")
    if any(part in {".git", ".cache", ".ssh", ".venv", "__pycache__"} for part in path.parts):
        raise ValueError(f"Local-only path {name!r}")
    if path.suffix in {".safetensors", ".gguf", ".pth", ".pem", ".key"}:
        raise ValueError(f"Unexpected weight or credential path {name!r}")


def package(output):
    prior_hashes = {}
    archive_members = {}
    workspace_members = {}
    manifest = {}
    with ExitStack() as stack:
        for basename in OLD_RELEASES:
            prior = PACKAGE_DIR / f"{basename}.zip"
            receipt = read_json(prior.with_suffix(".receipt.json"))
            expected = receipt["sha256" if basename == "course_handoff_mac_v2" else "archive_sha256"]
            assert fingerprint(prior) == expected
            prior_hashes[basename] = expected
            source = stack.enter_context(zipfile.ZipFile(prior))
            for member in source.infolist():
                name = member.filename
                if member.is_dir() or name in OLD_ZIPS or name in OLD_MANIFESTS:
                    continue
                verify_name(name)
                assert member.external_attr >> 28 != 0xA
                archive_members[name] = source
        for relative in FINAL_FILES:
            local = ROOT / relative
            for file in sorted(local.rglob("*")) if local.is_dir() else [local]:
                if file.is_dir():
                    continue
                assert file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(ROOT)
                name = file.relative_to(ROOT).as_posix()
                verify_name(name)
                workspace_members[name] = file
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=2) as final:
            for name in sorted(set(archive_members) | set(workspace_members)):
                file = workspace_members.get(name)
                source = file.open("rb") if file else archive_members[name].open(name)
                with source, final.open(name, "w", force_zip64=True) as target:
                    digest = hashlib.sha256()
                    for block in iter(lambda: source.read(4 << 20), b""):
                        digest.update(block)
                        target.write(block)
                manifest[name] = digest.hexdigest()
            final.writestr(MANIFEST, json.dumps({"files": manifest, "prior_release_sha256": prior_hashes}, indent=2) + "\n")


def verify_extract(archive, destination):
    with zipfile.ZipFile(archive) as source:
        manifest = json.loads(source.read(MANIFEST))["files"]
        assert sorted(source.namelist()) == sorted([MANIFEST, *manifest])
        for name in source.namelist():
            verify_name(name)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / MANIFEST).write_bytes(source.read(MANIFEST))
        for name, checksum in manifest.items():
            output = destination / name
            output.parent.mkdir(parents=True, exist_ok=True)
            with source.open(name) as entry, output.open("xb") as target:
                shutil.copyfileobj(entry, target, 4 << 20)
            assert fingerprint(output) == checksum, name
    return {"status": "PASS_EXTRACTED_ARCHIVE_HASHES", "archive_sha256": fingerprint(archive),
            "archive_bytes": archive.stat().st_size, "verified_files": len(manifest)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=OUTPUT)
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    if not args.archive.exists():
        package(args.archive)
    if args.extract_to:
        print(json.dumps(verify_extract(args.archive, args.extract_to), indent=2))
