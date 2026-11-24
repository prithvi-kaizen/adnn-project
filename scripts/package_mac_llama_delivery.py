"""Seal the local Llama extension beside the previously verified handoff."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from course_llm_audit import ROOT, load_json, save

FILES = (
    "output/course_fast_track/course_handoff_mac_v1.zip", "output/course_fast_track/course_handoff_mac_v1.receipt.json",
    "output/pdf/course_acl_mac_v2", "artifacts/exploratory/mac_llama_retrospective_v1",
    "artifacts/exploratory/mac_llamacpp_layers_v1", "artifacts/exploratory/mac_reviewer_interactive_v1",
    "docs/evidence/mac_engineering_v1", "docs/evidence/mac_llama_retrospective_v1", "docs/MAC_ENGINEERING_VALIDATION.md",
    "docs/MAC_LLAMA_RETROSPECTIVE_PROTOCOL.md", "docs/MAC_LLAMA_RETROSPECTIVE_RESULTS.md", "docs/MAC_LLAMA_HANDOFF.md",
    "docs/PHASE_STATUS.md", "docs/REQUIREMENTS_MATRIX.md", "docs/RISK_REGISTER.md", "docs/COURSE_SUBMISSION_GUIDE.md",
    "scripts/extract_mac_llama_features.cpp", "scripts/prepare_mac_llama_features.py", "scripts/run_mac_guarded_process.py",
    "scripts/run_mac_llama_probes.py", "scripts/run_mac_llama_stated.py", "scripts/verify_mac_llama_retrospective.py",
    "scripts/build_mac_acl_paper.py", "scripts/package_mac_llama_delivery.py", "scripts/check_cached_llama_layers.cpp",
    "scripts/check_mac_reviewer_interactive.py", "scripts/verify_mac_engineering.py", "paper/course_acl/mac_llama_probe.tex",
    "src/probes/row_space.py", "tests/test_probe_row_space.py", "tests/test_interactive_reviewer.py", "results/registry.jsonl",
)


def digest(handle):
    value = hashlib.sha256()
    for block in iter(lambda: handle.read(4 << 20), b""):
        value.update(block)
    return value.hexdigest()


def create(path):
    prior = ROOT / FILES[0]
    with prior.open("rb") as file:
        assert digest(file) == load_json(ROOT / FILES[1])["archive_sha256"]
    entries = []
    for name in FILES:
        candidate = ROOT / name
        if candidate.is_dir():
            entries.extend(p for p in candidate.rglob("*") if p.is_file())
        else:
            entries.append(candidate)
    checks = {}
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
        for file in sorted(entries):
            assert file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(ROOT)
            name = str(file.relative_to(ROOT))
            with file.open("rb") as source:
                checks[name] = digest(source)
            archive.write(file, name, compress_type=zipfile.ZIP_STORED if file == prior else zipfile.ZIP_DEFLATED)
        archive.writestr("LLAMA_HANDOFF_MANIFEST.json", json.dumps(checks, indent=2) + "\n")
    with zipfile.ZipFile(path) as archive:
        for name, expected in checks.items():
            with archive.open(name) as file:
                assert digest(file) == expected
    with path.open("rb") as package:
        save(path.with_suffix(".receipt.json"), {"status": "PASS_STREAMING_ARCHIVE_HASHES", "bytes": path.stat().st_size,
             "sha256": digest(package), "files": len(checks), "original_full_study_complete": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output.resolve()
    if not output.is_relative_to(ROOT / "output/course_fast_track"):
        parser.error("Use the local course handoff output tree")
    create(output)
