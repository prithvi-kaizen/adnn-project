"""Map every frozen requirement to the deadline plan, without claiming results."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUMAN = {7, 8, 28, 29, 88, 103}
PHYSICAL = {18, 22, 23, 24, 25, 26, 27, 30, 75, 76}
RESOURCE = {34, 35, 38, 39, 97, 104, 105}


def phase_ids(row):
    n = int(row['id'][1:])
    if n in {7, 8, 88}:
        return [1, 6]
    if n in {28, 29, 103}:
        return [2, 5, 6]
    if n in {5, 6, 21, 72, 92, 108, 109}:
        return [5]
    if n in {34, 35, 39, 104, 105}:
        return [3, 4]
    if n == 27:
        return [2, 4, 5]
    if n == 80:
        return [5]
    if n == 113:
        return [6]
    return [{1: 1, 2: 2, 3: 4, 4: 4, 5: 5, 6: 6}[row['phase']]]


def strategy(row):
    n = int(row['id'][1:])
    if n in HUMAN:
        return ('UNMET_WITHOUT_HUMAN_EVIDENCE',
                'Record not performed/unknown human contribution or reading evidence. Preserve the required forms/calculation code; no Codex/model judgment can satisfy the real-person criterion. No waiver has been received.')
    if row['priority'] == 'BONUS':
        return ('OPTIONAL_AFTER_CORE',
                'RQ4 controlled steering is the preferred bonus route only after core outputs and delivery reserve are secure. RQ5 waits for legitimate data/oracle feasibility. Mark NOT RUN if incomplete; no bonus-readiness claim from a demo.')
    if row['priority'] == 'NICE_TO_HAVE':
        return ('DEFER_UNLESS_CORE_COMPLETE',
                'ToolEmu adaptation, 14B and optional caution prompting cannot delay core experiments or packaging. Document omission; do not claim a run.')
    if n in PHYSICAL:
        return ('SCIENTIFIC_GATE_REQUIRED',
                'Retain the original two-environment target and exact/partial/horizon rules. Existing sandbox evidence is usable; physical completeness/action/partial-policy failures remain unknown. A sandbox-only pilot leaves the physical/joint/transfer part incomplete.')
    if n in RESOURCE:
        return ('RESOURCE_AND_RUNTIME_GATE',
                'Refresh actual allocated resources and access, pin revisions, run excluded-pilot smoke and measured projection. Prefer prescribed sequential native BF16 models. Small/quantized fallback is explicitly incomplete for the prescribed comparison.')
    if n == 37:
        return ('MODEL_CHANGE_CONDITIONAL',
                'Use prescribed families for original compliance. A reduced-course checkpoint or family change is recorded as a deviation, without invented instructor approval or an original-compliance claim.')
    if n == 52:
        return ('AUTOMATABLE_BUT_TARGET_UNMET',
                'Target 60 distinct validated R/I paired task families with safe-route goal witnesses. First verify existing 12 development instances and exclude partial-only pairs from the R/I gap by a prespecified label rule. Renaming instances does not fulfill 60 families.')
    if n == 78:
        return ('AUTOMATE_OR_REPORT_SHORTFALL',
                'Run seeds 1729/2718/31415 on grouped splits using shared cached features; lock all development choices before any test outcome. Fewer completed seeds must be explicit.')
    if n in {79, 80}:
        return ('AUTOMATE',
                'Compute registered whole-component/paired-family 1000-resample intervals and exact paired tests from actual outputs. Report independent N, missingness and undefined metrics.')
    if n in {82, 83, 87, 115}:
        return ('AUTOMATE_DELIVERY',
                'Produce and visually check the actual two-column 6–8-page report, required sections/citations and 15-minute presentation/demo. Label scripted/replayed material and all missing experiments.')
    if n in {84, 85, 86, 93, 110, 111, 112, 113, 114, 116, 117}:
        return ('AUTOMATE_AUDIT_AND_PACKAGE',
                'Build from saved data/results/source receipts; verify hashes and reproduction; expose limitations, absent human metrics and failed/null attempts. Do not publish/submit externally without instruction.')
    return ('AUTOMATE',
            'Implement or reuse the mapped artifact and run the original validation within the selected course scope. Report actual N/model/environment/control coverage; no simulated measurements or silent denominator changes.')


def build():
    requirements = json.loads((ROOT / 'docs/requirements.json').read_text())
    live = {r['id']: r for r in json.loads((ROOT / 'docs/evidence/requirements_status_causal_v1.json').read_text())}
    freeze = json.loads((ROOT / 'docs/evidence/protocol_freeze.json').read_text())
    assert len(requirements) == 117 and len({r['id'] for r in requirements}) == 117
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in freeze['sha256'].items())
    records = []
    for row in requirements:
        mode, action = strategy(row)
        records.append({
            'id': row['id'], 'source': row['source'], 'priority': row['priority'],
            'requirement': row['requirement'], 'original_phase': row['phase'],
            'course_phases': phase_ids(row), 'original_artifact': row['artifact'],
            'acceptance_test': row['validation'], 'deadline_policy': mode, 'deadline_action': action,
            'prior_development_status': live[row['id']]['status'],
            'prior_status_scope_limit': live[row['id']]['scope_limit'],
            'prior_evidence': live[row['id']]['evidence'], 'deadline_completion_status': 'NOT_YET_VERIFIED',
        })
    return {
        'scope': 'all_original_requirements_mapped_to_plan_not_completed', 'version': 'v1',
        'requirements_source_sha256': hashlib.sha256((ROOT / 'docs/requirements.json').read_bytes()).hexdigest(),
        'original_frozen_integrity': 'PASS', 'original_requirements_mapped': len(records),
        'priority_counts': dict(Counter(r['priority'] for r in records)),
        'deadline_policy_counts': dict(Counter(r['deadline_policy'] for r in records)),
        'human_evidence_unmet_ids': [r['id'] for r in records if r['deadline_policy'] == 'UNMET_WITHOUT_HUMAN_EVIDENCE'],
        'full_marks_guaranteed': False, 'human_requirement_waived': False,
        'original_assignment_completion_claimed': False, 'records': records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-markdown', type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_markdown.exists():
        raise SystemExit('Preserve prior versions; choose new outputs')
    report = build()
    lines = ['# Original requirements under the two-day plan', '',
             'All 117 original IDs are retained: 107 MUST, seven BONUS and three NICE_TO_HAVE. This is coverage of the plan, not 117 completed requirements or a marks prediction. Prior PASS/PARTIAL/BLOCKED statuses refer only to their limited development evidence recorded before this deadline decision. Every new course completion is still NOT_YET_VERIFIED. The original artifact/acceptance test remains visible even when a reduced course artifact cannot satisfy it.', '',
             'The researcher requests no human judgments. R007/R008/R028/R029/R088/R103 need genuine human evidence and cannot be completed by Codex alone; no instructor waiver is present. ScienceWorld/joint-data/transfer and two-family native experiments remain conditional. Optional and bonus items are retained and may be NOT RUN.', '',
             '| ID | Priority / source | Requirement | Course phase | Deadline policy and action | Original artifact / acceptance test | Prior development evidence and limit |',
             '|---|---|---|---|---|---|---|']
    for r in report['records']:
        values = [r['id'], r['priority'] + '; ' + r['source'], r['requirement'],
                  ','.join(map(str, r['course_phases'])), r['deadline_policy'] + ': ' + r['deadline_action'],
                  r['original_artifact'] + '; ' + r['acceptance_test'],
                  r['prior_development_status'] + ': ' + r['prior_evidence'] + '; ' + r['prior_status_scope_limit']]
        lines.append('| ' + ' | '.join(v.replace('|', '/') for v in values) + ' |')
    args.output_json.write_text(json.dumps(report, indent=2) + '\n')
    args.output_markdown.write_text('\n'.join(lines) + '\n')
    print('PASS: 117 unique original requirements mapped; human gaps retained; frozen hashes unchanged')


if __name__ == '__main__':
    main()
