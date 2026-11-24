"""Check the optional hand-drawn paper and preserve the verified measured figures."""

from pathlib import Path

from verify_course_submission import read, sha, verify

QA = Path("docs/evidence/course_handdrawn_v1/paper_qa.json")


def main():
    verify()
    receipt = read(QA)
    assert receipt["status"] == "PASS_HANDDRAWN_VARIANT_VISUAL_AND_SOURCE"
    assert receipt["pages_total"] == receipt["pages_visually_checked"] == 9
    assert receipt["main_pages"] == 8 and receipt["reference_pages"] == 1
    assert receipt["sketch_dimensions_px"][0] == 2 * receipt["sketch_dimensions_px"][1]
    assert receipt["new_model_experiment_calls"] == 0
    assert all(sha(Path(path)) == checksum for path, checksum in receipt["source_sha256"].items())
    print("PASS: hand-drawn Fig. 1, eight main pages plus references, and unchanged measured figures.")


if __name__ == "__main__":
    main()
