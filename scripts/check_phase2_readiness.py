"""Audit the current development artifacts without manufacturing a final freeze."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.annotation import validate_blank_package  # noqa: E402
from src.data.splits import components, validate_split  # noqa: E402
from src.data.validation import validate_records  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = ROOT / 'data/candidates/v5/sandbox_pairs.jsonl'
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    validation = validate_records(rows)
    for seed in (1729, 2718, 31415):
        manifest = json.loads((ROOT / f'data/splits/candidate_v5/candidate_split_{seed}.json').read_text())
        if manifest['dataset_sha256'] != hashlib.sha256(payload).hexdigest():
            raise ValueError('Split dataset hash mismatch')
        validate_split(rows, manifest)
    package = {p.name: p.read_text() for p in (ROOT / 'data/annotation/preview_v5').glob('*.csv')}
    annotation = validate_blank_package(package)
    final = validate_records(rows, recompute=False, require_final=True)
    report = {'scope': 'current_phase2_acceptance_audit_not_final_dataset',
              'sandbox_validation': validation, 'sandbox_sha256': hashlib.sha256(payload).hexdigest(),
              'conservative_components': len(set(components(rows).values())),
              'candidate_split_leakage_check': 'PASS', 'candidate_split_final_freeze': False,
              'annotation_package': annotation, 'required_real_manual_review': 100,
              'genuine_completed_double_annotations': 0,
              'certified_scienceworld_labels': 0, 'final_gate': 'PARTIAL',
              'final_acceptance_check': final['status'],
              'final_rejection_reasons': dict(Counter(e.split(': ', 1)[1] if e.startswith('row ') else e
                                                      for e in final['errors'])),
              'dataset_frozen': False,
              'blockers': ['ScienceWorld full-state equality and physical partial policy uncertified',
                           'Instructor decision needed before scoped physical-state amendment',
                           'At least 200 more eligible examples and meaningful required physical coverage',
                           'Final joint dataset/splits and real independent 100-case human checkpoint']}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('final_gate', 'certified_scienceworld_labels', 'dataset_frozen', 'blockers')}, indent=2))
    if validation['status'] != 'PASS' or final['status'] != 'FAIL':
        raise SystemExit('Unexpected acceptance state')


if __name__ == '__main__':
    main()
