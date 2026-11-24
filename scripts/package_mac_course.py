"""Package the validated Mac overlay and the unchanged verified baseline archive."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from course_llm_audit import ROOT, load_json, save, sha

OVERLAY = (
    "artifacts/exploratory/mac_safer_behavior_v1", "artifacts/exploratory/mac_reviewer_transitions_v1",
    "artifacts/exploratory/mac_cached_smoke_v1", "artifacts/exploratory/mac_cached_smoke_v2",
    "artifacts/exploratory/mac_cached_smoke_v3", "artifacts/exploratory/mac_reviewer_fixtures_v1",
    "docs/evidence/mac_safer_behavior_v1", "docs/evidence/mac_phases_2_to_4_v1", "docs/evidence/course_mac_cached_v1",
    "docs/evidence/course_acl_v1/paper_qa.json", "docs/evidence/course_llm_annotation_local_v1/preflight.json",
    "output/pdf/course_acl_mac_v1", "output/pdf/course_acl/latex_source.zip", "output/presentation/course_acl",
    "configs/course_llm_annotation_local_v1.json", "configs/course_llm_annotation_local_v2.json",
    "configs/course_mac_cached_smoke_v1.json", "configs/course_mac_cached_smoke_v2.json",
    "configs/course_mac_cached_smoke_v3.json", "configs/mac_safer_behavior_v1.json",
    "scripts/run_mac_safer_behavior.py", "scripts/summarize_mac_safer_behavior.py", "scripts/audit_mac_safer_candidates.py",
    "scripts/check_mac_reviewer_fixtures.py", "scripts/check_mac_reviewer_transitions.py", "scripts/build_mac_acl_paper.py",
    "scripts/benchmark_course_mac_cached.py", "scripts/run_course_llm_audit_local.py", "scripts/verify_mac_matched_smokes.py",
    "scripts/package_mac_course.py",
    "scripts/check_mac_continuation_scoring.py", "src/models/continuation_scoring.py", "src/data/safer_route_candidates.py",
    "paper/course_acl/mac_development.tex", "paper/course_acl/mac_discussion.tex",
    "tests/test_mac_budget.py", "tests/test_continuation_scoring.py", "results/registry.jsonl",
    "docs/MAC_SAFER_ROUTE_RESULTS.md", "docs/MAC_COURSE_HANDOFF.md", "docs/MAC_ONLY_EXECUTION_PLAN.md",
    "docs/MAC_PHASES_2_TO_4.md", "docs/FULL_STUDY_IMPLEMENTATION_PLAN.md", "docs/CONTRIBUTIONS.md",
    "docs/PHASE_STATUS.md", "docs/REQUIREMENTS_MATRIX.md", "docs/RISK_REGISTER.md", "docs/COURSE_SUBMISSION_GUIDE.md",
)
BASE = ROOT / "output/course_fast_track/course_handoff_v4.zip"


def package(path):
    if sha(BASE) != load_json(BASE.with_suffix(".receipt.json"))["archive_sha256"]:
        raise ValueError("Baseline archive hash mismatch")
    files = [BASE, BASE.with_suffix(".receipt.json")]
    for entry in OVERLAY:
        source = ROOT / entry
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_dir():
            files.extend(p for p in source.rglob("*") if p.is_file() and not p.name.startswith("."))
        else:
            files.append(source)
    manifest = {}
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for file in sorted(set(files)):
            if file.is_symlink() or not file.resolve().is_relative_to(ROOT):
                raise ValueError("Unsafe archive input")
            name = str(file.relative_to(ROOT))
            manifest[name] = sha(file)
            archive.write(file, arcname=name, compress_type=zipfile.ZIP_STORED if file == BASE else zipfile.ZIP_DEFLATED)
        archive.writestr("MAC_HANDOFF_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None and len(archive.namelist()) == len(manifest) + 1
        for name, expected in manifest.items():
            with archive.open(name) as handle:
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
                assert digest.hexdigest() == expected
    save(path.with_suffix(".receipt.json"), {"status": "PASS_MAC_HANDOFF_CRC_AND_HASHES", "archive": str(path.relative_to(ROOT)),
         "archive_sha256": sha(path), "archived_files": len(manifest), "size_bytes": path.stat().st_size,
         "baseline_sha256": manifest[str(BASE.relative_to(ROOT))], "full_original_study_complete": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    target = parser.parse_args().output.resolve()
    if not target.is_relative_to(ROOT / "output/course_fast_track"):
        parser.error("Use the local course handoff output tree")
    package(target)
