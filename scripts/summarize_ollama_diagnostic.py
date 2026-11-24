"""Independently rescore the fixed format diagnostic from raw outcomes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_exploratory_ollama import parser_diagnostic  # noqa: E402
from scripts.summarize_exploratory_ollama import rate  # noqa: E402
from src.data.pilot import scenarios  # noqa: E402


def summarize(directory):
    rows = [json.loads(x) for x in (directory / 'outcomes.jsonl').read_text().splitlines()]
    lookup = {s.scenario_id: s for s in scenarios()}
    for row in rows:
        if row['scenario_id'] and row['status'] == 'completed':
            if parser_diagnostic(lookup[row['scenario_id']], row['text']) != row['parser']:
                raise ValueError('Parser score disagrees with raw output')
    report = {'scope': 'exploratory_engineering_not_main_study', 'conditions': {}, 'canaries': [],
              'assigned_fixture_calls': 16, 'observed_fixture_calls': sum(r['scenario_id'] is not None for r in rows),
              'assigned_canaries': 2, 'observed_canaries': sum(r['scenario_id'] is None for r in rows),
              'raw_outcomes_sha256': hashlib.sha256((directory / 'outcomes.jsonl').read_bytes()).hexdigest(),
              'metadata_sha256': hashlib.sha256((directory / 'metadata.json').read_bytes()).hexdigest(),
              'bootstrap_seed': 9001, 'bootstrap_resamples': 1000, 'no_world_tools_executed': True}
    for condition in ('original_raw', 'explicit_raw', 'explicit_raw_schema', 'explicit_chat_schema'):
        subset = [r for r in rows if r['condition'] == condition]
        if len(subset) != 4:
            report['conditions'][condition] = {'status': 'incomplete', 'observed': len(subset), 'assigned': 4}
            continue
        report['conditions'][condition] = {key: rate(subset, key) for key in
                                           ('json_valid', 'grammar_valid', 'legal_action')}
    for row in rows:
        if row['scenario_id'] is None:
            compliant = row['status'] == 'completed' and json.loads(row['text']) == {'probe': 'ok'}
            if compliant != row['canary_compliant']:
                raise ValueError('Canary score mismatch')
            report['canaries'].append({'condition': row['condition'], 'compliant': compliant,
                                      'raw_text': row['text']})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(ROOT / 'artifacts/exploratory/ollama_diagnostic_v2')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: {m: v[m]['successes'] for m in v} for k, v in report['conditions'].items()}, indent=2))


if __name__ == '__main__':
    main()
