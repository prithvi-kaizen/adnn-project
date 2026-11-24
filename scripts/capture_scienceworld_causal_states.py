"""Save exact proposed target values for the already tested three door traces."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scienceworld_adapter.causal import ProposedDoorReplay  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Preserve existing evidence')
    original = json.loads((ROOT / 'data/scienceworld/causal_proposal_v1_attempt4/report.json').read_text())
    report = {'scope': 'raw_proposed_target_receipts_not_certified_state_or_dataset',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runtime_source_sha256': original['source_sha256'], 'records': [], 'eligible_physical_labels': 0}
    for row in original['door_inverses']:
        env = ProposedDoorReplay(row['task'], row['variation'])
        try:
            opening, closing = row['actions']
            for name, prefix in (('initial', ()), ('opened', (opening,)), ('closed', (opening, closing))):
                state = env.replay(prefix)
                assert state.digest() == row[name + '_target_hash']
                assert not state.complete
                report['records'].append({'task': row['task'], 'variation': row['variation'],
                    'phase': name, 'prefix': prefix, 'payload': state.payload, 'sha256': state.digest(),
                    'complete': state.complete, 'incompleteness_reason': state.incompleteness_reason})
        finally:
            env.close()
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: nine saved target payloads reproduce the original inverse-trace hashes; no labels')


if __name__ == '__main__':
    main()
