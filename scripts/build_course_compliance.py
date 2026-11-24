"""Produce an explicit requirement-by-requirement audit, preserving the freeze."""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
audit = {}


def put(numbers, status, evidence):
    for number in numbers:
        key = f"R{number:03}"
        assert key not in audit
        audit[key] = {"status": status, "evidence_and_limit": evidence}


put([1, 81, 93, 94, 95], "PASS", "COURSE_RED_TEAM_AUDIT.md; COURSE_PHASE4_AUDIT.md; actual nulls, parsing failures and distinct denominators retained; all tools isolated")
put([2, 3, 4], "PARTIAL", "Native Mac CPU course records and statistics exist for one 0.5B family and exposed sandbox tasks; prescribed two-family/physical/final-family scope incomplete")
put([5, 92], "PARTIAL", "COURSE_FIRST_DECISION_AMENDMENT.md and verified 102-cell diagnostic/strength curve; one decision on six exposed versions; full 15-turn RQ4 NOT RUN")
put([6, 109], "BLOCKED", "RQ5 NOT RUN: executable-game licensing/recoverability oracle not certified; no fabricated uncertainty result")
put([7, 8, 28, 29, 88, 103], "BLOCKED", "CONTRIBUTIONS.md; real reading, 100 independent human double annotations, adjudication/agreement and contribution attestations absent; no waiver or synthetic substitute")
put([9], "PASS", "LITERATURE_REVIEW.md and writing_primary_sources.json; primary-source metadata verified; report uses seven verified references")
put([10], "PASS", "Dated NOVELTY_MATRIX.md and source audit; bounded overlap search, no first-ever claim; no exact duplicate asserted or administrator/instructor message sent")
put(range(11, 18), "PASS", "src/sandbox and sandbox/agent tests; in-memory files, mail, wallet, read/edit/delete/send/transfer/ask_user with no real capabilities")
put([18, 26, 75], "BLOCKED", "SCIENCEWORLD_AMENDMENT_FOR_REVIEW.md; source/replay/search diagnostics exist, causal-state approval/completeness unverified; no physical labels or transfer results")
put([19, 21], "PARTIAL", "LICENSE_PROVENANCE.md; native Qwen Apache assets verified, own sandbox data; ByteSized permission/executable-game requirements incomplete; local package not publicly released")
put([20, 36, 77], "BLOCKED", "Optional ToolEmu adaptation, 14B comparison and caution-prompt variant NOT RUN; no expected-result cells")
put([22, 23, 24, 25, 27], "PARTIAL", "Finite sandbox state/replay, exhaustive h=1/2/3 exact/fee-only partial labels pass; complete physical causal-state/action certification absent")
put([30], "PARTIAL", "COURSE_DATA_CARD.md: 1,800 sandbox examples, zero certified physical; original >=2,000 across both environments unmet")
put([31, 32, 33, 41, 42, 43, 44, 45], "PASS", "COURSE_PHASE2_AUDIT.md and v2 validation/freeze: h3 630/390/780; JSONL schema; exact grouped 20% tests; 810 pairs/90% membership; deterministic full-state sandbox mechanisms")
put([34], "PARTIAL", "One native Qwen 0.5B family measured. Separate Ollama engineering runs do not supply the prescribed native two-family comparison")
put([35], "BLOCKED", "Prescribed Qwen2.5-7B then Llama-3.1-8B BF16 study NOT RUN; queue/storage/data gates incomplete; no substitution claim")
put([37], "PARTIAL", "Reduced course checkpoint authorized by researcher and disclosed before outcome use; no instructor-approved fulfillment of original model specification recorded")
put([38, 39, 40], "PASS", "COURSE_PHASE3_AUDIT.md and active execution config: measured cache/activation budget, sequential native CPU, Transformers/sklearn, pinned greedy seeds/prompts/raw outputs")
put([46, 47, 104, 105], "PASS", "features_manifest.json, independent features verification and native-helper tests: all 1,800 single forwards, action spans/padding/raw-vs-normalized sites, exact revisions/software/template hashes")
put([48, 49], "PASS", "180-condition selection/evaluation locks: training-only scaling, L2 linear regression, binary/three-way, four C values, development-only layer selection, convergence checks")
put([50, 55, 56, 57, 58, 67, 68, 69, 70, 71], "PASS", "All-layer saved native course metrics and layer figure; majority/action-only/ten shuffles/fixed-verb/layer0 controls present. Generalization remains one small sandbox condition")
put([51, 59, 60, 107], "PASS", "1,800 raw stated outputs and same-ID joins; invalids incorrect, invalid kappa category, confusion/agreement/coverage saved; not human kappa")
put([52], "PARTIAL", "24 exposed versions of 12 pairs/four templates; nine R/I pairs/three eligible templates. Original 60 independent held-out families and some recoverable safe-route witnesses unmet")
put([53, 54, 61, 62, 63, 64, 65, 66], "PASS", "24 full native course episodes, max15/two-invalid abort, all replayed; pre-state I oracle, caution, goal, over-caution, sensitivity and exact K-A with honest degeneracy and denominators")
put([72, 108], "PASS", "Native helper tests plus diagnostic: three norm-matched random directions, exact raw block binding, all six zero token identities, hook removal after each condition; no full-RQ4 completion claim")
put([73, 74, 78], "PASS", "Last/mean, pair-only/all and three grouped split allocations executed; test overlap disclosed; descriptive mean/sample SD saved, never pooled as independent replications")
put([76], "PARTIAL", "Four sandbox ablation axes executed; mandatory bidirectional physical transfer is missing and not replaced by extra sandbox conditions")
put([79], "PARTIAL", "1,000 whole-component/template draws and metric-specific missingness rules; selected-condition metrics and all-layer accuracy CIs saved. Other raw per-layer F1/BA/AUROC lack layerwise CIs; missing two-model scope remains")
put([80], "PASS", "Exact paired discordant counts and p=1 for zero discordance; joint-template intervals and dependence caveat; no independent-60-family or completed two-model Holm claim")
put([82, 83, 115], "PASS", "Eight-page two-column IEEE report, editable LaTeX, seven verified BibTeX entries; abstract/intro/related work/design/results/limits/conclusion; every final PDF page visually inspected")
put([84, 114], "PARTIAL", "COURSE_DELIVERY_GUIDE.md, source locks/software pins, no-download integrity/replay smoke and historical rerun snapshot; full original gated pipeline remains blocked; large vectors/weights outside compact package")
put([85], "PARTIAL", "JSONL/course data card with hashes, counts, splits and explicit missing human agreement exists; original joint final dataset and human checkpoint incomplete")
put([86, 89, 90, 91], "PASS", "Saved predictions/raw model traces, generated publication vectors, comparison/degenerate-gap plot and 16-row ablation table; every value traces to saved records")
put([87], "PARTIAL", "Editable 12-slide talk/900-second notes, untimed bonus backup and working two-version saved-trace demo prepared; actual presenter rehearsal/delivery not performed")
put([96, 97, 98, 99], "PASS", "Original planning/freeze, source manifest, actual resource checks or explicit scheduling/storage blockers and current risk register; six original frozen files unchanged")
put([100, 101], "PASS", "Pure sandbox transitions, snapshots/replay and conservative search; incomplete ScienceWorld searches remain unknown, never I")
put([102, 106], "PARTIAL", "Course provenance/duplicate grouping/train-only normalization/dev selection pass; physical completeness and bidirectional source/destination transfer validation absent")
put([110], "PASS", "C4 independent records audit, 720 point checks, 24 explicit-row primary bootstrap checks, joint-template intervals, 102 first-decision replays and raw-generation association checks")
put([111], "PASS", "COURSE_RED_TEAM_AUDIT.md and qualitative_audit.json: three alternatives and all four predefined strata inspected, counts 3/0/3/0; absent examples not invented; action proxies explicit")
put([112], "PASS", "Vector PDF/SVG figures generated from locked results; readable final axes/legends and visual-QA source-hash receipts")
put([113], "PASS", "This full 117-ID evidence audit preserves original acceptance targets and exposes partial/blocked items; no marks guarantee")
put([116], "PASS", "C5/C6 gate receipts, hash/size manifest, tests, local handoff and explicit original/human gaps; submission checklist remains a real-author task")
put([117], "PASS", "No public push, upload, email or submission; local package only; credentials/weights/user-owned untracked annotation directory excluded")

requirements = json.loads((ROOT / "docs/requirements.json").read_text())
assert set(audit) == {r["id"] for r in requirements} and len(audit) == 117
result = {"date": "2026-11-17", "interpretation": "Item-level evidence; PASS does not waive cross-cutting original scope or complete the assignment", "full_original_compliance": False,
          "counts": dict(Counter(r["status"] for r in audit.values())),
          "requirements": [{**r, "final_audit": audit[r["id"]]} for r in requirements]}
(ROOT / "docs/evidence/course_phase5_20261117/final_compliance.json").write_text(json.dumps(result, indent=2)+"\n")
lines = ["# Final original-requirement audit", "", "17 November 2026. All 117 IDs are retained. PASS means the stated item has inspectable evidence; it does not waive the cross-cutting two-family, physical-data, sample-size or human requirements. PARTIAL identifies a working reduced artifact with a remaining original shortfall. BLOCKED includes explicit NOT RUN optional work. These counts are not a mark prediction.", "", "Full original-assignment compliance: **false**. Original research gates remain PARTIAL. The course package is a separate reduced-scope deliverable.", "", "The original frozen requirements are unchanged; the matrix retains its historical rows with a current checkpoint. Current evidence takes precedence over the historical development statuses in COURSE_FAST_TRACK_REQUIREMENTS.md. Missing layerwise non-accuracy intervals are disclosed under R079 rather than advertised as complete.", "", "| ID | Priority | Status | Evidence and limit |", "|---|---|---|---|"]
lines += [f"| {r['id']} | {r['priority']} | {audit[r['id']]['status']} | {audit[r['id']]['evidence_and_limit']} |" for r in requirements]
(ROOT / "docs/COURSE_FINAL_COMPLIANCE.md").write_text("\n".join(lines)+"\n")
print(json.dumps(result["counts"]))
