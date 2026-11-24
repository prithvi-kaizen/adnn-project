"""Run existing tiny random native/hook checks on MPS without CPU fallback."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_native_smoke import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Preserve prior attempts')
    import torch
    started = monotonic()
    report = {'scope': 'tiny_random_MPS_engineering_not_pretrained_or_steering_research',
              'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': platform.python_version(), 'platform': platform.platform(),
              'MPS_available': torch.backends.mps.is_available(),
              'CPU_fallback_enabled': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1',
              'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'tokenizers')},
              'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
                  'scripts/check_course_mps.py', 'scripts/check_native_smoke.py', 'src/models/native_smoke.py')},
              'weights_downloaded': False, 'pretrained_model_loaded': False}
    try:
        assert report['MPS_available'] and not report['CPU_fallback_enabled']
        report['records'] = run('mps', 'float32')
        torch.mps.synchronize()
        report['mps_current_allocated_bytes_after_test'] = torch.mps.current_allocated_memory()
        report['mps_driver_allocated_bytes_after_test'] = torch.mps.driver_allocated_memory()
        report['status'] = 'PASS'
    except Exception as error:
        report.update(status='FAIL', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    report['elapsed_seconds'] = monotonic() - started
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'MPS_available', 'CPU_fallback_enabled',
                                           'elapsed_seconds', 'weights_downloaded')}, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
