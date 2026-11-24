"""Verify Phase I receipts offline plus a fresh local disk-capacity guard."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/course_phase1_20261115'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    freeze = read(ROOT / 'docs/evidence/protocol_freeze.json')
    assert all(sha(ROOT / p) == expected for p, expected in freeze['sha256'].items())
    mapping = read(ROOT / 'docs/evidence/course_fast_track_requirement_map_v1.json')
    assert mapping['original_requirements_mapped'] == len({r['id'] for r in mapping['records']}) == 117
    assert not mapping['human_requirement_waived'] and not mapping['full_marks_guaranteed']
    baseline_log = ROOT / 'docs/evidence/course_fast_track_unittest_20261115.log'
    log = baseline_log.read_text()
    assert 'Ran 101 tests' in log and '\nOK\n' in log and 'FAILED' not in log and 'skipped=' not in log
    # The historical receipt covers its original files. Later modules and their
    # tests need their own gate receipts rather than invalidating that history.
    baseline_paths = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', 'a052bec', '--', 'src', 'tests'], cwd=ROOT, text=True).splitlines()
    unchanged = subprocess.check_output(['git', 'diff', '--name-only', 'a052bec', '--', *baseline_paths],
                                        cwd=ROOT, text=True)
    assert unchanged == '', 'Existing test receipt invalidated by core/test edits'
    data = read(ROOT / 'docs/evidence/course_fast_track_data_check_20261115.json')
    assert data['sandbox_validation']['status'] == 'PASS' and data['sandbox_validation']['rows'] == 1800
    assert data['sandbox_validation']['verified_minimal_pair_members'] == 1620
    mps = read(EVIDENCE / 'mps_native_hooks.json')
    assert mps['status'] == 'PASS' and mps['MPS_available'] and not mps['CPU_fallback_enabled']
    assert len(mps['records']) == 2 and not mps['pretrained_model_loaded']
    for model in mps['records']:
        assert model['device'] == 'mps' and model['precision'] == 'float32'
        assert all(model[key] for key in ('zero_strength_logits_identical', 'zero_strength_greedy_ids_identical',
                                         'only_last_position_changes', 'hook_cleanup_on_exception'))
    for p, expected in mps['source_sha256'].items():
        assert sha(ROOT / p) == expected
    native = read(ROOT / 'docs/evidence/model_storage_metadata_20261101.json')['models']
    access = {}
    for name in ('qwen', 'llama'):
        report = read(EVIDENCE / f'{name}_access_cpu.json')
        assert report['job_id'] == '35998' and report['access_verified']
        assert not report['weights_downloaded'] and not report['token_or_redirect_url_recorded']
        assert report['source_sha256'] == sha(ROOT / 'scripts/hpc/check_model_access.py')
        assert report['revision'] == native[report['model']]['revision']
        for actual, expected in zip(report['shards'], native[report['model']]['primary_safetensors']):
            assert actual['status'] == 'ACCESS_VERIFIED' and actual['bytes'] == expected['bytes']
            assert actual['etag'] == expected['lfs']['sha256']
        access[report['model']] = True
    home = int((EVIDENCE / 'home_allocated_cpu.txt').read_text().split()[0])
    reserves = read(ROOT / 'docs/evidence/staged_storage_budget_20261101.json')[
        'staged_components_planning_reserves_not_measurements']
    projections = {}
    for name, specification in native.items():
        total = home + sum(s['bytes'] for s in specification['primary_safetensors']) + sum(reserves.values())
        projections[name] = {'projected_bytes_including_reserve': total,
                             'margin_beyond_reserved_bytes': 50_000_000_000 - total,
                             'fit_under_written_50GB_policy': total <= 50_000_000_000}
        assert total <= 50_000_000_000
    scheduler = (EVIDENCE / 'scheduler_final.txt').read_text()
    assert '35997|CANCELLED' in scheduler and '35998|COMPLETED|00:00:06|0:0|dgxa' in scheduler
    assert '2026-11-17T07:24:10' in (EVIDENCE / 'gpu_resource_options.txt').read_text()
    fallback = read(EVIDENCE / 'fallback_model_metadata.json')
    assert fallback['status'] == 'PASS' and fallback['capacity_passed']
    assert fallback['source_sha256'] == sha(ROOT / 'scripts/check_course_fallback_metadata.py')
    assert fallback['license'] == 'apache-2.0' and not fallback['weights_downloaded']
    assert fallback['repo'] == 'Qwen/Qwen2.5-0.5B-Instruct'
    assert fallback['revision'] == '7ae557604adf67be50417f59c2c2f167def9a775'
    assert all(f['access_verified_by_HEAD'] for f in fallback['files'])
    assert sum(f['bytes'] for f in fallback['files']) == fallback['checkpoint_and_tokenizer_bytes']
    free = shutil.disk_usage(ROOT).free
    assert free >= fallback['required_free_bytes_before_download'], 'Local disk margin no longer passes'
    toolchain = read(EVIDENCE / 'tooling/report_toolchain.json')
    assert toolchain['status'] == toolchain['visual_QA_status'] == 'PASS'
    assert toolchain['page_count'] == 1 and toolchain['canonical_text_checks'] == 'PASS'
    assert toolchain['source_sha256'] == sha(ROOT / 'scripts/check_course_report_toolchain.py')
    for name, expected in toolchain['outputs_sha256'].items():
        assert sha(EVIDENCE / 'tooling' / name) == expected
    assert len(read(EVIDENCE / 'citation_refresh.json')['records']) == 6
    config = read(ROOT / 'configs/course_fast_track.json')
    selected = config['selected_course_route']
    assert (selected['model'], selected['revision'], selected['device'], selected['planned_precision']) == (
        fallback['repo'], fallback['revision'], 'mps', 'float32')
    assert not selected['pretrained_model_has_run'] and not selected['trust_remote_code']
    bonus = config['bonus_RQ4']
    assert bonus['target_requested'] and bonus['strengths'] == [-2, -1, 0, 1, 2]
    assert bonus['random_direction_seeds'] == [811, 812, 813]
    assert bonus['distinct_conditions_per_version'] == 5 + 3 * 4
    assert not bonus['bonus_completion_claimed'] and bonus['require_real_checkpoint_zero_strength_identity']
    paths = [baseline_log, ROOT / 'configs/course_fast_track.json',
             ROOT / 'requirements-course-tools-lock.txt', ROOT / 'scripts/verify_course_phase1.py']
    paths += [p for p in EVIDENCE.rglob('*') if p.is_file()]
    return {
        'scope': 'fast_track_Phase_I_readiness_not_original_assignment_completion',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'course_gate_C1': 'PASS',
        'original_Gate1': 'PARTIAL', 'original_Gate2': 'PARTIAL',
        'selected_route': selected, 'native_model_performance_verified': False,
        'baseline_tests': {'passed': 101, 'skipped': 0, 'baseline_source_unchanged_since_receipt': True,
                           'later_source_additions_require_separate_test_receipts': True},
        'Mac_tiny_native_MPS_and_hook_tests': 'PASS', 'pretrained_steering_tested': False,
        'plot_IEEE_compile_render_and_visual_QA': 'PASS',
        'DGX_CPU_job': {'job_id': 35998, 'state': 'COMPLETED', 'elapsed_seconds': 6},
        'DGX_full_A100_job': {'job_id': 35997, 'state': 'CANCELLED_BEFORE_RUNNING',
                              'scheduler_estimated_start_raw': '2026-11-17T07:24:10',
                              'fresh_GPU_memory_verified': False},
        'MIG_scheduling_test_only': {'job_number_displayed': 35999, 'actual_allocation': False,
                                    'scheduler_estimated_start_raw': '2026-12-11T21:45:46'},
        'both_original_models_access_verified': access, 'home_allocated_bytes': home,
        'remote_staged_storage_projections': projections,
        'fallback_checkpoint_and_tokenizer_bytes': fallback['checkpoint_and_tokenizer_bytes'],
        'local_free_bytes_at_gate_check': free,
        'local_required_free_bytes_including_reserves': fallback['required_free_bytes_before_download'],
        'refresh_capacity_before_actual_download': True,
        'bonus_RQ4_target': True, 'bonus_model_episode_projection_still_required': True,
        'new_pretrained_model_runs': 0, 'weight_files_downloaded': 0, 'human_judgments_created': 0,
        'sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Choose a new verification path')
    report = verify()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS C1: source/test/data receipts, MPS hooks, exact fallback budget, report toolchain and bonus plan')
    print('Original gates remain partial. No pretrained weights or results; Phase III benchmark still required.')


if __name__ == '__main__':
    main()
