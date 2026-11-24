"""Revalidate saved sandbox pilot records without loading any model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.pilot import simulator_source_hash  # noqa: E402
from src.data.validation import validate_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()
    try:
        rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    report = validate_records(
        rows, require_final=args.require_final, expected_source_hash=simulator_source_hash(ROOT)
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
