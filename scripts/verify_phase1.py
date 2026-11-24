"""Check Phase 1 planning integrity without packages, models, or network access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    required = (
        "MASTER_PLAN", "REQUIREMENTS_MATRIX", "LITERATURE_REVIEW", "NOVELTY_MATRIX",
        "EXPERIMENT_PROTOCOL", "COMPUTE_BUDGET", "DECISION_LOG", "PHASE_STATUS",
        "RISK_REGISTER", "LICENSE_PROVENANCE", "MODEL_SELECTION",
    )
    for name in required:
        path = ROOT / "docs" / f"{name}.md"
        assert path.is_file() and path.stat().st_size > 100, path
        assert "\u2014" not in path.read_text(), f"Em dash in {path}"

    manifest = json.loads((ROOT / "docs/evidence/input_manifest.json").read_text())
    for name, entry in manifest["inputs"].items():
        assert digest(ROOT / name) == entry["sha256"], f"Source changed: {name}"
    extracted = (ROOT / "docs/evidence/ASSIGNMENT_EXTRACT.txt").read_text()
    assert "[body:116]" in extracted and "[body:000]" in extracted

    config = json.loads((ROOT / "configs/project.yaml").read_text())
    assert config["execution"]["allow_model_download"] is False
    assert config["execution"]["allow_real_tools"] is False
    assert config["hpc"]["capacity_approved"] is False
    assert config["splits"]["seeds"] == [1729, 2718, 31415]
    assert config["behavior"]["final_pairs"] >= 60
    assert config["statistics"]["bootstrap_resamples"] == 1000

    requirements = json.loads((ROOT / "docs/requirements.json").read_text())
    assert len({row["id"] for row in requirements}) == len(requirements)
    for row in requirements:
        assert all(row.get(key) for key in (
            "source", "priority", "phase", "requirement", "artifact", "validation", "status"
        )), row
        assert row["priority"] in {"MUST", "BONUS", "NICE_TO_HAVE"}
        assert row["phase"] in range(1, 7)
        assert row["id"] in (ROOT / "docs/REQUIREMENTS_MATRIX.md").read_text()

    registry = [json.loads(line) for line in
                (ROOT / "configs/experiment_registry.jsonl").read_text().splitlines()]
    assert len({row["experiment_id"] for row in registry}) == len(registry)
    assert {row["rq"] for row in registry} == {f"RQ{i}" for i in range(1, 6)}
    assert all(row["status"] == "planned" and row["outputs"] == [] for row in registry)
    assert all(row["runtime_seconds"] is None for row in registry)

    evidence = json.loads((ROOT / "docs/evidence/external_sources.json").read_text())
    assert evidence["citation_audit"]["auditCounts"]["matched"] == 12
    assert len(json.loads(evidence["files"][4]["content"])) == 144

    freeze = json.loads((ROOT / "docs/evidence/protocol_freeze.json").read_text())
    assert freeze["scope"] == "pilot_only"
    assert freeze["final_test_outcomes_seen"] is False
    for name, expected in freeze["sha256"].items():
        assert digest(ROOT / name) == expected, f"Frozen file changed: {name}"
    print(f"PASS: {len(requirements)} requirement mappings, {len(registry)} planned "
          "experiment families, source/citation/configuration/freeze integrity.")
    print("This verifies planning consistency only. GPU capacity, runtime correctness, "
          "annotations and experimental results have separate gates.")


if __name__ == "__main__":
    main()
