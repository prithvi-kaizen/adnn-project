"""Verify the revised local paper and replay-only package with the standard library."""

import subprocess
import sys
from pathlib import Path

from verify_course_improvement_handoff import ROOT, read, sha

QA = Path("docs/evidence/course_submission_v1/paper_qa.json")
FIGURES = Path("figures/course_submission/v1")


def verify():
    for command in ("verify_course_delivery.py", "verify_course_improvement_handoff.py"):
        subprocess.run([sys.executable, "scripts/" + command], cwd=ROOT, check=True)
    receipt = read(QA)
    assert receipt["status"] == "PASS_REVISED_PAPER_VISUAL_CONTENT_AND_SOURCE"
    assert receipt["pages_total"] == receipt["rendered_pages_checked"] == 9
    assert receipt["main_pages"] == 8 and receipt["reference_pages"] == 1
    assert sha(Path(receipt["paper"])) == receipt["paper_sha256"]
    assert all(sha(Path(path)) == expected for path, expected in receipt["source_sha256"].items())
    manifest = read(FIGURES / "manifest.json")
    assert all(sha(Path(path)) == expected for path, expected in manifest["source_sha256"].items())
    assert all(sha(FIGURES / path) == expected for path, expected in manifest["files_sha256"].items())
    assert manifest["svg_text_is_editable"] and not manifest["external_content_reproduced"]
    assert receipt["measured_baseline_figures_unchanged"] and receipt["new_model_calls"] == 0
    print("PASS: eight main pages plus references, source hashes, two vector figures, and verified baseline.")


if __name__ == "__main__":
    verify()
