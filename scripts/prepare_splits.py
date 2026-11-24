"""Save provisional split feasibility, keeping final holdout unfrozen."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.splits import candidate_split, validate_split  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = args.dataset.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in (1729, 2718, 31415):
        report = candidate_split(rows, seed)
        report["dataset_sha256"] = hashlib.sha256(payload).hexdigest()
        if "indices" in report:
            validate_split(rows, report)
        path = args.output_dir / f"candidate_split_{seed}.json"
        content = json.dumps(report, indent=2) + "\n"
        if path.exists() and path.read_text() != content:
            raise SystemExit(f"Changed split requires a new version: {path}")
        path.write_text(content)
        print(json.dumps({k: v for k, v in report.items() if k not in {"indices", "components"}}))


if __name__ == "__main__":
    main()
