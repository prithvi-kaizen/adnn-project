"""Pin and budget the public small native fallback with HEAD requests only."""

import argparse
import hashlib
import json
import logging
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Preserve earlier attempts')
    logging.disable(logging.CRITICAL)
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    repo = 'Qwen/Qwen2.5-0.5B-Instruct'
    report = {'scope': 'fallback_revision_access_and_capacity_no_weight_download',
              'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'repo': repo,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'weights_downloaded': False, 'pretrained_model_loaded': False}
    try:
        api = HfApi(token=False)
        revision = api.model_info(repo).sha
        info = api.model_info(repo, revision=revision, files_metadata=True)
        assert info.sha == revision and not info.gated
        license_name = info.card_data.to_dict().get('license')
        assert license_name == 'apache-2.0'
        by_name = {s.rfilename: s for s in info.siblings}
        allowlist = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                     'special_tokens_map.json', 'model.safetensors')
        files = []
        for name in allowlist:
            if name not in by_name:
                assert name == 'special_tokens_map.json'
                continue
            s = by_name[name]
            metadata = get_hf_file_metadata(hf_hub_url(repo, name, revision=revision), token=False, timeout=15)
            assert metadata.commit_hash == revision and metadata.size == s.size
            if s.lfs:
                expected = s.lfs.sha256
                assert metadata.etag == expected
            else:
                expected = s.blob_id
                assert metadata.etag == expected
            files.append({'path': name, 'bytes': s.size, 'etag': metadata.etag,
                          'lfs_sha256': s.lfs.sha256 if s.lfs else None,
                          'access_verified_by_HEAD': True})
        small_assets = {}
        for name in ('config.json', 'LICENSE'):
            with urllib.request.urlopen(hf_hub_url(repo, name, revision=revision), timeout=20) as response:
                payload = response.read(100_001)
            assert len(payload) <= 100_000
            small_assets[name] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
            if name == 'config.json':
                config = json.loads(payload)
                report['config'] = {k: config[k] for k in (
                    'architectures', 'model_type', 'num_hidden_layers', 'hidden_size',
                    'torch_dtype', 'vocab_size', 'tie_word_embeddings')}
                assert config['model_type'] == 'qwen2'
        total = sum(f['bytes'] for f in files)
        free = shutil.disk_usage('.').free
        reserve, outputs, temporary = 2_147_483_648, 402_653_184, 67_108_864
        required = total + reserve + outputs + temporary
        report.update(revision=revision, license=license_name, files=files,
            small_metadata_assets_fetched=small_assets, checkpoint_and_tokenizer_bytes=total,
            disk_free_bytes=free, disk_reserve_bytes=reserve, planned_output_allowance_bytes=outputs,
            planned_temporary_allowance_bytes=temporary, required_free_bytes_before_download=required,
            margin_after_all_reserves_bytes=free-required, capacity_passed=free >= required,
            candidate_device='mps', candidate_precision='float32',
            precision_is_not_prescribed_BF16=True, original_two_family_comparison_fulfilled=False,
            model_performance_and_peak_memory_verified=False,
            status='PASS' if free >= required else 'BLOCKED_STORAGE')
    except Exception as error:
        report.update(status='FAIL', error_type=type(error).__name__)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'repo', 'weights_downloaded')}))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
