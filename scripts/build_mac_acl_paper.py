"""Build a versioned ACL paper from the verified portable source and Mac results."""

import argparse
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from course_llm_audit import ROOT, load_json, save, sha

BASE = ROOT / "docs/evidence/course_acl_v1/paper_qa.json"
SUPPLEMENT = ROOT / "paper/course_acl/mac_development.tex"
DISCUSSION = ROOT / "paper/course_acl/mac_discussion.tex"
LLAMA_SUPPLEMENT = ROOT / "paper/course_acl/mac_llama_probe.tex"


def replace_once(template, anchor, revised):
    if template.count(anchor) != 1:
        raise ValueError(f"Missing or duplicate portable-source anchor: {anchor!r}")
    return template.replace(anchor, revised)


def build(target, rows, llama, review=None, steering=None):
    reference = load_json(BASE)
    base = ROOT / reference["source_zip"]
    if sha(base) != reference["source_zip_sha256"]:
        raise ValueError("Portable ACL baseline changed")
    target.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="adnn-acl-mac-") as tmp:
        stage = Path(tmp).resolve()
        with zipfile.ZipFile(base) as archive:
            assert archive.testzip() is None
            assert all((stage / name).resolve().is_relative_to(stage) for name in archive.namelist())
            archive.extractall(stage)
        manuscript = stage / "report.tex"
        manuscript.write_text(replace_once(manuscript.read_text(), "These results characterize one model",
            "Separate development runs with two cached quantized models also show\n"
            "that a successful finish can coexist with an irreversible action.\n"
            "The probe results characterize one model"))
        results = stage / "results.tex"
        results.write_text(results.read_text() + "\n\\input{mac_development}\n")
        shutil.copyfile(DISCUSSION, stage / "discussion.tex")
        if llama:
            evidence = load_json(llama)
            assert evidence["probe_condition_count"] == 180 and evidence["stated_unique"]["N"] == 96
            matched = [rec for rec in evidence["primary_probe"] if rec["condition"]["binary"] and
                       rec["condition"]["subset"] == "minimal_pairs"]
            assert len(matched) == len(evidence["stated"]) == 3
            table = []
            for rec, judged in zip(matched, evidence["stated"]):
                assert judged["seed"] == rec["condition"]["seed"]
                probe_ba, lexical_ba = (rec[key]["metrics"]["balanced_accuracy"]["estimate"]
                                        for key in ("probe", "lexical"))
                pair = rec["probe"]["both_pair_endpoints_correct"]
                line = (f"{judged['seed']} & {probe_ba:.1%} & {lexical_ba:.1%} & {pair['correct_pairs']}/{pair['N']} & "
                        f"{judged['correct']}/{judged['N']}")
                table.append(line.replace("%", r"\%") + r" \\")
            source = replace_once(LLAMA_SUPPLEMENT.read_text(), r"\input{mac_llama_rows}", "\n".join(table))
            (stage / "mac_llama_probe.tex").write_text(source)
            results.write_text(results.read_text() + "\n\\input{mac_llama_probe}\n")
            discussion = stage / "discussion.tex"
            shortened = replace_once(discussion.read_text(), "The probe uses one native float32 checkpoint. The two quantized models\n"
                "have separate development traces without probe features.", "Native and quantized probes use different numerical conditions and sample sizes.")
            shortened = replace_once(shortened,
                "The selected probe gained no balanced accuracy over the action-only\n"
                "baseline on matched states at the registered threshold. Some continuous\n"
                "scores shifted within pairs. The result therefore concerns the selected\n"
                "readout and decision rule.\n\n"
                "Correct labels and cautious actions separate at the task level. In the\n"
                "cached runs, confirmation often preceded an irreversible action. Both models\n"
                "had a zero aggregate gap, although Llama's correct judgment and increased\n"
                "caution occurred on different pairs. The code and replay package preserves\n"
                "the individual outcomes.",
                "The native probe tied action text on matched states despite some\n"
                "within-pair score shifts. In the cached runs, confirmation often\n"
                "preceded irreversible execution. Llama's correct judgment and\n"
                "increased caution occurred on different pairs, yielding a zero\n"
                "aggregate gap. The replay package preserves those outcomes.")
            shortened = replace_once(shortened, "Codex assisted with code, analysis, writing, and typesetting. Individual\n"
                "human contribution and reading attestations remain pending.",
                "Codex assisted with code, analysis and writing.")
            discussion.write_text(shortened)
        if review:
            audit = load_json(review)
            assert audit["status"] == "COMPLETE_SINGLE_BLINDED_SANDBOX_REVIEW" and audit["reviewer_count"] == 1
            agreement = audit["oracle_vs_review"]["cohens_kappa"]["estimate"]
            invalid = len(audit["bad_witness_cases"])
            assert invalid == 2
            line = (f"One blinded Codex reviewer matched the simulator on {audit['label_correct']}/"
                    f"{audit['raw_case_count']} cases ($\\kappa={agreement:.3f}$). "
                    "Two false recoverability claims failed witness replay.")
            results.write_text(replace_once(results.read_text(), "The blinded 100-case model-review\npacket received no returns.", line))
            discussion = stage / "discussion.tex"
            discussion.write_text(replace_once(discussion.read_text(), "The\n"
                "instructor-approved model review remains pending. Neither cached reviewer\n"
                "qualified, and the blind packet stayed unopened.", "The sandbox audit used one LLM reviewer.\n"
                "A second review and the physical audit remain open."))
        if steering:
            evidence = load_json(steering)
            assert evidence["status"] == "PASS_EXPLORATORY_FULL_EPISODE_REPLAY"
            assert len(evidence["conditions"]) == 9 and evidence["zero_exact_traces"] == 6
            outcomes = {(c["N"], c["caution"], c["goal"], c["irreversible"], c["invalid_abort"],
                         c["goal_with_finish"], c["world_trace_changed"]) for c in evidence["conditions"]}
            assert outcomes == {(6, 0, 2, 1, 6, 0, 0)}
            assert all(c["token_trace_changed"] for c in evidence["conditions"] if c["strength"])
            body = replace_once(results.read_text(),
                "All 102 assigned cells replay successfully. Six use the shared\n"
                "cached zero baseline. The diagnostic makes 106 fresh requests,\n"
                "including zero checks and timing calls. Every direction and strength\n"
                "produces zero caution, two legal calls, and two reached goals out of\n"
                "six, with one irreversible execution. The other four decisions are\n"
                "rejected. The paired differences and conditional template intervals\n"
                "are zero (Figure~\\ref{fig:steering}). The full 15-turn steering\n"
                "sweep and the next-state uncertainty test were not run.",
                "All 102 cells replayed, including six reused zeros. Each condition had 0/6 cautious versions, "
                "2/6 legal calls and goals, and 1/6 irreversible executions (Figure~\\ref{fig:steering}).")
            extension = (f"We repeated the five strengths through all turns in {evidence['cells']} development episodes "
                f"({evidence['turns']} calls), with the seed-811 random control. Zero reproduced every baseline token. "
                "Both directions changed tokens, but every condition retained 0/6 cautious versions, 2/6 goals and "
                "1/6 irreversible executions. All episodes reached the two-invalid-call limit.")
            results.write_text(replace_once(body, "\\subsection{Additional Development Tasks}",
                                              extension + "\n\n\\subsection{Additional Development Tasks}"))
            discussion = stage / "discussion.tex"
            updated = replace_once(discussion.read_text(),
                "Native and quantized probes use different numerical conditions and sample sizes. The probe splits\n"
                "overlap, and several estimates have few independent components. The steering\n"
                "test covers the first decision. The two-family BF16 study, certified\n"
                "ScienceWorld data, and the 60-family behavior evaluation remain open. The sandbox audit used one LLM reviewer.\n"
                "A second review and the physical audit remain open.",
                "The probe splits overlap, and some estimates have few independent components. Native and quantized runs "
                "use different sample sizes. Steering used exposed tasks with invalid-call stops. One LLM reviewed "
                "sandbox labels. The BF16 comparison, certified ScienceWorld labels, held-out behavior and steering, "
                "and the next-state uncertainty test remain open.")
            updated = replace_once(updated,
                "\\section*{Use of AI Assistance}\n\nCodex assisted with code, analysis and writing.", "")
            updated = replace_once(updated, "One LLM reviewed sandbox labels.",
                                   "Codex assisted with code, analysis, writing and a single blinded label audit.")
            discussion.write_text(updated)
        supplement = replace_once(SUPPLEMENT.read_text(), r"\input{mac_development_rows}", rows.read_text().rstrip())
        (stage / "mac_development.tex").write_text(supplement)
        shutil.copyfile(rows, stage / "mac_development_rows.tex")
        paths = sorted(path for path in stage.rglob("*") if path.is_file())
        with zipfile.ZipFile(target / "latex_source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                member = zipfile.ZipInfo(str(path.relative_to(stage)), date_time=(2026, 11, 20, 0, 0, 0))
                archive.writestr(member, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
        output = stage / "build"
        env = dict(os.environ, SOURCE_DATE_EPOCH="1795132800")
        proc = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                               f"-outdir={output}", "report.tex"], cwd=stage, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (target / "latexmk.log").write_bytes(proc.stdout)
        proc.check_returncode()
        shutil.copyfile(output / "report.pdf", target / "report.pdf")
        warnings = [line for line in (output / "report.log").read_text().splitlines()
                    if any(term in line for term in ("Warning:", "Overfull", "Underfull"))]
        save(target / "build_receipt.json", {"base_sources_sha256": sha(base), "rows_sha256": sha(rows),
             "supplement_sha256": sha(SUPPLEMENT), "discussion_sha256": sha(DISCUSSION),
             "paper_sha256": sha(target / "report.pdf"),
             "sources_sha256": sha(target / "latex_source.zip"), "final_log_warnings": warnings,
             "retrospective_verification_sha256": sha(llama) if llama else None,
             "codex_review_sha256": sha(review) if review else None,
             "full_episode_steering_sha256": sha(steering) if steering else None,
             "full_original_study_complete": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--llama", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--steering", type=Path)
    args = parser.parse_args()
    build(args.output.resolve(), args.rows.resolve(), args.llama, args.review, args.steering)
