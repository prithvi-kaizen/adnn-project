"""Render live acceptance status without modifying frozen requirements.json."""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-output', type=Path,
                        default=ROOT / 'docs/evidence/requirements_status_certification_v2.json')
    args = parser.parse_args()
    requirements = json.loads((ROOT / 'docs/requirements.json').read_text())
    status = {}
    for row in requirements:
        if row['phase'] >= 3:
            status[row['id']] = ('BLOCKED', 'docs/PHASE_STATUS.md', 'Primary work awaits accepted data/access and earlier phase gates')
        else:
            status[row['id']] = ('PARTIAL', 'docs/PHASE2_REQUIREMENTS_AUDIT.md', 'Current development scope only; final acceptance remains gated')
    def assign(ids, state, evidence, detail):
        for number in ids:
            status[f'R{number:03d}'] = state, evidence, detail
    assign([7, 8], 'BLOCKED', 'docs/CONTRIBUTIONS.md', 'Real teammate reading/summary attestations absent')
    assign([9, 10], 'PASS', 'docs/LITERATURE_REVIEW.md; docs/NOVELTY_MATRIX.md', 'Bounded primary-source identity/overlap audit; refresh before manuscript')
    assign([19], 'PARTIAL', 'docs/DATA_CARD.md', 'Terms audited; no public release; final model access and release license pending')
    assign([37], 'PASS', 'docs/evidence/protocol_freeze.json', 'No primary checkpoint/family substitution; quantized runs explicitly exploratory')
    assign([38, 97], 'PARTIAL', 'docs/INFRASTRUCTURE_UPDATE_20261101.md', 'Typed full-A100 route and staged storage verified; fresh native model peak check pending')
    assign([96, 98, 99], 'PASS', 'docs/MASTER_PLAN.md; docs/RISK_REGISTER.md; docs/evidence/protocol_freeze.json', 'Planning preregistration and frozen hashes verified; does not pass every phase gate')
    assign([11, 12, 13, 14, 15, 16, 17, 41, 42, 43, 44, 95, 100, 101], 'PASS',
           'src/sandbox/; tests/; data/candidates/v5/validation.json', 'Implemented sandbox and conservative unknown handling; no physical-certification claim')
    assign([18, 22, 23, 24, 25, 26], 'PARTIAL',
           'docs/SCIENCEWORLD_CAUSAL_STATE_AUDIT.md; data/scienceworld/causal_proposal_v1_attempt4/report.json; tests/test_causal_proposal.py',
           '34-file audit, six engine adversaries and guarded h1/2/3 diagnostics; zero certified physical families/labels; causal closure, complete grammar and partial policy unresolved')
    assign([28, 30], 'BLOCKED', 'docs/SCIENCEWORLD_AMENDMENT_FOR_REVIEW.md; docs/ANNOTATION_WORKFLOW.md',
           'Physical labels, accepted joint >=2000 pool and genuine independent 100-case review absent')
    assign([31, 32, 45, 102], 'PARTIAL', 'data/candidates/v5/; data/splits/candidate_v5/; docs/DATA_CARD.md',
           '1800 sandbox candidates, 90% minimal-pair members, exact provisional splits; no final joint freeze')
    assign([33], 'PASS', 'data/candidates/v5/validation.json', 'Mandatory JSONL schema passes for sandbox candidates; final joint dataset absent')
    assign([103], 'PARTIAL', 'src/data/annotation.py; scripts/import_annotations.py; docs/ANNOTATION_WORKFLOW.md; output/pdf/annotation_handoff_pilot_v1/',
           '100-case independent blank pilot packets and tested kappa/CI importer; final joint sample and real originals/adjudication pending')
    assign([20], 'BLOCKED', 'docs/DATA_CARD.md', 'Optional adaptation deferred behind compulsory dataset validity; no paid emulator')
    assign([27, 39, 46, 104, 105, 108], 'PARTIAL', 'docs/NATIVE_BF16_SMOKE_PROTOCOL.md; docs/evidence/native_smoke_cpu_bf16_v1.json',
           'Tiny random two-family mechanics and actual pinned tokenizer alignment tested; no pretrained activations or primary outcome claim')
    assign([40, 54], 'PARTIAL', 'docs/OLLAMA_ENGINEERING_RESULTS.md; docs/OLLAMA_DIAGNOSTIC_v2_RESULTS.md',
           'Exploratory prompts, settings and failures preserved; primary behavior/caution scoring pending')
    assign([53], 'PASS', 'src/agent/loop.py; tests/test_agent.py', 'Strict one-call loop, <=15 slots and two-invalid abort tested; final episodes remain gated')
    assign([1, 79, 81, 94, 110, 113], 'PARTIAL', 'results/registry.jsonl; docs/PHASE_STATUS.md; scripts/summarize_ollama_diagnostic.py',
           'Registered engineering evidence/failures and independent summaries; primary scientific audit awaits results')
    assign([84, 85, 86, 93, 114, 116], 'PARTIAL', 'docs/REPRODUCIBILITY.md; docs/DATA_CARD.md; docs/PHASE_STATUS.md',
           'Development artifacts and honest limitations exist; final research deliverables remain incomplete')
    assign([117], 'PASS', 'docs/DATA_CARD.md; docs/PHASE_STATUS.md', 'No public push, release or submission performed')
    lines = ['# Requirements matrix', '',
             'Live acceptance audit, 2 November 2026, continuing from e7f2bed while preserving the later annotation handoff. Frozen requirement definitions remain unchanged in requirements.json. PASS applies only to the stated scope. BLOCKED includes phase dependencies, human judgment and external authorization. This matrix does not claim final submission acceptance.', '',
             '| ID | Source | Priority | Phase | Requirement | Planned artifact | Validation | Status | Current evidence / limit |',
             '|---|---|---|---|---|---|---|---|---|']
    current = []
    for row in requirements:
        state, evidence, detail = status[row['id']]
        values = [row['id'], row['source'], row['priority'], str(row['phase']), row['requirement'],
                  row['artifact'], row['validation'], state, evidence + ': ' + detail]
        lines.append('| ' + ' | '.join(str(v).replace('|', '/') for v in values) + ' |')
        current.append({'id': row['id'], 'phase': row['phase'], 'status': state, 'evidence': evidence, 'scope_limit': detail})
    lines += ['', '## Counts by phase', '', '| Phase | PASS | PARTIAL | BLOCKED |', '|---|---:|---:|---:|']
    for phase in range(1, 7):
        count = Counter(r['status'] for r in current if r['phase'] == phase)
        lines.append(f"| {phase} | {count['PASS']} | {count['PARTIAL']} | {count['BLOCKED']} |")
    (ROOT / 'docs/REQUIREMENTS_MATRIX.md').write_text('\n'.join(lines) + '\n')
    subset = ['# Phase 2 requirement audit', '', 'Current acceptance by frozen ID. Scope limits apply; Gate 2 remains PARTIAL.', '', '| ID | Status | Evidence / limit |', '|---|---|---|']
    subset += ['| ' + r['id'] + ' | ' + r['status'] + ' | ' + r['evidence'] + ': ' + r['scope_limit'] + ' |' for r in current if r['phase'] == 2]
    (ROOT / 'docs/PHASE2_REQUIREMENTS_AUDIT.md').write_text('\n'.join(subset) + '\n')
    args.evidence_output.write_text(json.dumps(current, indent=2) + '\n')


if __name__ == '__main__':
    main()
