"""Certify draft safer-route designs and screen for links to earlier cases."""

import argparse
from pathlib import Path

from audit_course_behavior_readiness import HISTORY, ROOT, read, read_rows, sha
from course_llm_audit import save

from src.data.behavior_readiness import joint_exposure_graph
from src.data.safer_route_candidates import certify


def audit():
    directory = Path("data/course_improvement/v1/paid_recovery_split_drafts")
    design = Path("docs/evidence/course_improvement_v1/i2_behavior_design/verification.json")
    development = Path("docs/evidence/course_improvement_v1/i2_development/behavior_readiness.json")
    expansions = (Path("data/course_improvement/v1/sandbox_expansion/sandbox_additions.jsonl"),
                  directory / "paid_recovery_additions.jsonl")
    history = [row for path in (*HISTORY, *expansions) for row in read_rows(path)]
    history.extend(read(design)["tasks"])
    history.extend(read(development)["development_fixture"])
    candidates, evidence = certify()
    assignments = joint_exposure_graph(history, candidates)["candidate_assignments"]
    components = {member["group"] for member in assignments.values() if not member["quarantined"]}
    sources = (*HISTORY, *expansions, design, development, Path(__file__).relative_to(ROOT),
               Path("src/data/safer_route_candidates.py"), Path("src/sandbox/core.py"), Path("src/sandbox/labeling.py"),
               Path("src/data/behavior_readiness.py"), Path("src/data/splits.py"))
    return {"status": "PASS_SCRIPTED_CANDIDATES_ONLY", "historical_rows_screened": len(history),
            "unlinked_draft_components": len(components), "new_final_families": 0, "holdout_calls": 0, "model_calls": 0,
            "candidate_assignments": assignments, "tasks": candidates, "certificates": evidence,
            "source_sha256": {str(path): sha(path) for path in sources}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    save(parser.parse_args().output, audit())
