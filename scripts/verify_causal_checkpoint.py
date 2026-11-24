"""Offline verification of measured development evidence; Gate 2 stays partial."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict:
    freeze = read('docs/evidence/protocol_freeze.json')
    assert all(sha(ROOT / p) == v for p, v in freeze['sha256'].items())
    artifacts = []
    for attempt in range(1, 5):
        name = f'data/scienceworld/causal_proposal_v1_attempt{attempt}/report.json'
        r = read(name)
        suffix = '_source_complete.tar.gz' if attempt == 2 else '_source.tar.gz'
        archive = f'artifacts/source/causal_proposal_v1_attempt{attempt}' + suffix
        with tarfile.open(ROOT / archive) as tar:
            for p, expected in r['source_sha256'].items():
                assert hashlib.sha256(tar.extractfile(p).read()).hexdigest() == expected, (attempt, p)
                if attempt == 4:
                    assert sha(ROOT / p) == expected, p
        assert r['eligible_physical_labels'] == 0 and not r['amendment_applied']
        artifacts.extend([name, archive])
    r = read('data/scienceworld/causal_proposal_v1_attempt4/report.json')
    assert r['diagnostic_checks_pass'] and not r['failures']
    assert r['jar_sha256'] == 'e77b0fee7d68abe3ca5b12e57d86e2bfe7603f20c5200da0b0f085939faeb465'
    fixtures = {f['name']: f for f in r['fixtures']}
    assert len(fixtures) == 6 and all(f['expected_checks_pass'] for f in fixtures.values())
    plant = fixtures['plant_counter_alias_rejected']['arms']
    assert plant[0]['before'] == plant[1]['before']
    assert not plant[0]['water_deleted'] and plant[1]['water_deleted']
    assert all(a['unsupported'] and not a['state_complete'] for a in plant)
    for name in ('omitted_combustion_point_detected', 'omitted_phase_name_map_detected'):
        a, b = fixtures[name]['arms']
        assert a['before'] == b['before'] and a['direct'] != b['direct'] and a['after'] != b['after']
    a, b = fixtures['water_plus_cup_external_thermal_alias']['arms']
    assert a['projection_before'] == b['projection_before']
    assert a['water_temperature_after'] == 20.0 and b['water_temperature_after'] == 20.2565
    a, b = fixtures['water_only_thermal_scope_fails']['arms']
    assert a['water_only'] == b['water_only'] and a['with_container'] != b['with_container']
    assert a['water_temperature_after'] == 20.0 and b['water_temperature_after'] == 22.85
    a, b = fixtures['unexported_drain_context_rejected']['arms']
    assert a['before'] == b['before'] and a['water_parent_after'] != b['water_parent_after']
    assert a['drain_record'] != b['drain_record'] and all(x['drain_record']['unsupported'] for x in (a, b))
    assert len(r['replays']) == 9
    for replay in r['replays']:
        assert len(replay['samples']) == 3
        assert all(s == replay['samples'][0] for s in replay['samples'])
    assert len(r['door_inverses']) == 3
    assert all(x['target_inverse_matches'] and x['candidate_changes_target'] and not x['strict_tree_inverse_matches']
               and x['label'] == 'unknown' for x in r['door_inverses'])
    receipts_name = 'data/scienceworld/causal_proposal_v1_attempt4/target_state_receipts.json'
    receipts = read(receipts_name)
    assert len(receipts['records']) == 9 and receipts['eligible_physical_labels'] == 0
    assert receipts['runtime_source_sha256'] == r['source_sha256']
    assert sha(ROOT / 'scripts/capture_scienceworld_causal_states.py') == receipts['source_sha256']
    for receipt in receipts['records']:
        assert hashlib.sha256(receipt['payload'].encode()).hexdigest() == receipt['sha256']
        original = next(x for x in r['door_inverses'] if (x['task'], x['variation']) ==
                        (receipt['task'], receipt['variation']))
        assert receipt['sha256'] == original[receipt['phase'] + '_target_hash']
        assert not receipt['complete'] and receipt['incompleteness_reason']
    artifacts.append(receipts_name)
    assert {(s['task'], s['horizon']) for s in r['searches']} == {
        (t, h) for t in ('boil', 'freeze', 'chemistry-mix') for h in (1, 2, 3)}
    counts = {}
    for search in r['searches']:
        assert search['label'] == 'unknown' and not search['search_exhausted']
        assert not search['state_export_complete'] and not search['legal_action_space_complete']
        assert not search['partial_policy_complete'] and not search['dataset_eligible']
        assert search['duplicate_states_skipped'] == 0 and search['depth_reached'] == 1
        assert len(search['candidate_target_matches']) == 1
        reason = search['censor_reason']
        counts[reason] = counts.get(reason, 0) + 1
    assert counts == {'search_censored': 4, 'search_frontier_censored': 2, 'replay_execution_not_certified': 3}
    trials = [t for e in r['enumeration_checks'] for t in e.get('self_pour_trials', [])]
    assert len(trials) == 4
    for trial in trials:
        assert trial['ticks_before'] == trial['ticks_after']
        assert trial['history_after'] - trial['history_before'] == len(trial['history_classes']) == 5
    source = read('docs/evidence/scienceworld_causal_source_v2.json')
    assert len(source['files']) == 34 and len(source['material_declared_fields']) == 14
    assert source['material_direct_reader_covers_all_base_fields'] and source['handler_coverage_equal']
    assert not source['full_action_grammar_certified'] and not source['causal_closure_certified']
    assert len(source['registered_handlers']) == len(source['enumerated_handlers']) == 27
    data = read('docs/evidence/phase2_readiness_causal_v1.json')
    assert data['sandbox_validation']['status'] == 'PASS' and data['sandbox_validation']['rows'] == 1800
    assert data['sandbox_validation']['verified_minimal_pair_members'] == 1620
    assert data['candidate_split_leakage_check'] == 'PASS' and not data['dataset_frozen']
    assert data['certified_scienceworld_labels'] == 0 and data['final_gate'] == 'PARTIAL'
    native = read('docs/evidence/native_smoke_causal_cpu_bf16_v1.json')
    assert not native['weights_downloaded'] and len(native['records']) == 2
    for model in native['records']:
        assert model['precision'] == 'bfloat16' and model['device'] == 'cpu'
        assert all(model[k] for k in ('zero_strength_logits_identical', 'zero_strength_greedy_ids_identical',
                                     'only_last_position_changes', 'hook_cleanup_on_exception'))
    handoff_root = ROOT / 'output/pdf/annotation_handoff_pilot_v1'
    handoff = json.loads((handoff_root / 'handoff_manifest.json').read_text())
    assert handoff['n_per_person'] == 100 and not handoff['contains_automatic_answers']
    assert all(sha(handoff_root / p) == h for p, h in handoff['outputs'].items())
    assert all(sha(ROOT / handoff['source_packet'] / p) == h for p, h in handoff['source_files'].items())
    requirements = read('docs/evidence/requirements_status_causal_v1.json')
    assert len(requirements) == 117 and all(x['status'] in ('PASS', 'PARTIAL', 'BLOCKED') for x in requirements)
    log = (ROOT / 'docs/evidence/causal_proposal_local_unittest_v3.log').read_text()
    assert 'Ran 101 tests' in log and '\nOK\n' in log and 'FAILED' not in log and 'skipped=' not in log
    artifacts += ['docs/evidence/scienceworld_causal_source_v2.json',
                  'docs/evidence/phase2_readiness_causal_v1.json',
                  'docs/evidence/native_smoke_causal_cpu_bf16_v1.json',
                  'docs/evidence/requirements_status_causal_v1.json',
                  'docs/evidence/causal_proposal_local_unittest_v3.log',
                  'docs/evidence/causal_setup_receipts_v1.json',
                  'docs/SCIENCEWORLD_CAUSAL_STATE_AUDIT.md', 'docs/SCIENCEWORLD_AMENDMENT_FOR_REVIEW.md']
    return {'scope': 'offline_engineering_checkpoint_not_scientific_gate_completion',
            'frozen_integrity': 'PASS', 'engine_adversarial_fixtures': 6, 'replay_prefix_settings': 9,
            'repeats_per_setting': 3, 'guarded_searches': 9, 'search_unknown_counts': counts,
            'unit_tests_passed': 101, 'sandbox_candidates': 1800, 'minimal_pair_members': 1620,
            'annotation_handoff_integrity': 'PASS', 'native_tiny_cpu_bf16': 'PASS',
            'certified_physical_task_families': [], 'eligible_physical_labels': 0,
            'instructor_approval': 'PENDING', 'methodology_amendment_applied': False,
            'gate1': 'PARTIAL', 'gate2': 'PARTIAL', 'weights_downloaded': False,
            'sha256': {p: sha(ROOT / p) for p in artifacts},
            'artifact_bytes': {p: (ROOT / p).stat().st_size for p in artifacts}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Choose a new verification output')
    result = verify()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: causal evidence/source archives, guards, dataset, annotation packets, tiny BF16 and frozen hashes')
    print('Gate 2 PARTIAL; zero certified physical families/labels; instructor approval pending')


if __name__ == '__main__':
    main()
