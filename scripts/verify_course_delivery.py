"""Standard-library, no-download integrity check and replay of saved outcomes."""

import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.course_metrics import audit_course_episode  # noqa: E402


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    h = hashlib.sha256()
    with (ROOT / path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    for name, expected in read("docs/evidence/protocol_freeze.json")["sha256"].items():
        assert sha(name) == expected, name
    data = "data/course_fast_track/v2/"
    for name, entry in read(data+"freeze_manifest.json")["files"].items():
        assert sha(data+name) == entry["sha256"]
        assert (ROOT / (data+name)).stat().st_size == entry["bytes"]
    gate = read("docs/evidence/course_phase4_20261115/gate_c4.json")
    summary_path = "results/course_fast_track/native_mac_cpu_v1/core_summary.json"
    assert gate["status"] == "PASS_C4_REDUCED_SCOPE" and gate["summary_sha256"] == sha(summary_path)
    stats = "results/course_fast_track/native_mac_cpu_v1/statistics.json"
    verified = read("docs/evidence/course_phase5_20261117/statistics_verification.json")
    assert verified["statistics_sha256"] == sha(stats) and verified["point_metric_checks"] == 720
    assert verified["primary_intervals_independently_recomputed"] == 24
    for directory in ("figures/course_fast_track/publication_v1", "figures/course_fast_track/first_decision_steering_v2"):
        manifest = read(directory+"/manifest.json")
        entries = manifest.get("files_sha256", {r["file"]: r["sha256"] for r in manifest.get("outputs", [])})
        for name, expected in entries.items():
            assert sha(directory+"/"+name) == expected, name
    tasks = {t["id"]: t for t in map(json.loads, (ROOT / (data+"development_tasks.jsonl")).read_text().splitlines())}
    behavior = "artifacts/course_fast_track/native_mac_cpu_v1/behavior/"
    episodes = read(behavior+"report.json")["episodes"]
    assert len(episodes) == 24 and len({e["id"] for e in episodes}) == 24
    for entry in episodes:
        audit = audit_course_episode(tasks[entry["id"]], read(behavior+entry["episode_file"]))
        assert audit == entry["audit"]
    diagnostic_path = "results/course_fast_track/native_first_decision_steering_v1/summary.json"
    diagnostic = read(diagnostic_path)
    assert diagnostic["status"] == "PASS_INDEPENDENT_FIRST_DECISION_REPLAY" and not diagnostic["complete_full_RQ4"]
    for name, expected in diagnostic["source_sha256"].items():
        # The original analysis recorded __file__ as an absolute provenance path.
        # Check the packaged source, never fall back to the producing machine.
        source = Path(name)
        if source.is_absolute():
            assert source.parent.name == "scripts" and source.name == "summarize_course_first_decision.py"
            source = Path("scripts") / source.name
        assert not source.is_absolute() and ".." not in source.parts
        assert sha(source) == expected, name
    cells = list(map(json.loads, (ROOT / "artifacts/course_fast_track/native_first_decision_steering_v1/cells.jsonl").read_text().splitlines()))
    assert len(cells) == 102
    for cell in cells:
        assert audit_course_episode(tasks[cell["id"]], cell["episode"]) == cell["audit"]
    qa = read("docs/evidence/course_phase5_20261117/visual_qa.json")
    assert qa["status"] == "PASS_CODEX_VISUAL_INSPECTION" and qa["paper_pages"] == 8
    assert sha(qa["paper"]) == qa["paper_sha256"]
    assert sha("paper/course_fast_track/report.pdf") == qa["paper_sha256"]
    deck = "output/presentation/course_fast_track/presentation_v2.pptx"
    presentation = read("docs/evidence/course_phase5_20261117/presentation_validation.json")
    assert sha(deck) == presentation["finalSha256"] == qa["presentation_sha256"]
    assert presentation["firstPartyImport"]["passed"] and presentation["presentationLayout"]["finding_count"] == 0
    notes = read("output/presentation/course_fast_track/speaker_notes.json")
    assert len(notes) == 13 and sum(n["seconds"] for n in notes) == 900 and notes[-1]["seconds"] == 0
    ns = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
    core = read(summary_path)
    rows = [next(r for r in core["probe_conditions"] if r["id"] == f"s{s}_h3_binary_last_all_ordinary") for s in (1729, 2718, 31415)]
    expected = [[450, 525, 630, 255, 255, 390, 1095, 1020, 780],
                [float(f"{r[key]['accuracy']:.4f}") for key in ("selected_metrics", "lexical", "majority") for r in rows]]
    with zipfile.ZipFile(ROOT / deck) as archive:
        names = archive.namelist()
        assert len([n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")]) == 13
        for index, values in enumerate(expected, 1):
            xml = ET.fromstring(archive.read(f"ppt/slides/charts/chart{index}.xml"))
            actual = [float(v.text) for series in xml.findall(".//c:ser", ns)
                      for v in series.findall("./c:val/c:numRef/c:numCache/c:pt/c:v", ns)]
            assert len(actual) == 9 and all(abs(a-b) < 1e-10 for a, b in zip(actual, values)), (index, actual)
    compliance = read("docs/evidence/course_phase5_20261117/final_compliance.json")
    assert len(compliance["requirements"]) == 117 and not compliance["full_original_compliance"]
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_sandbox.py"], cwd=ROOT, check=True)
    print("PASS: six original freezes, course data hashes, 720/24 statistics receipt, 24 full and 102 first-decision replays, 8-page PDF hash, 13-slide package/chart source values, 900-second notes and 117-item audit.")
    print("No model inference, download, human annotation, native PowerPoint test or full-original-study completion is claimed.")


if __name__ == "__main__":
    main()
