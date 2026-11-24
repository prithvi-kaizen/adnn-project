"""Generate and independently validate the 24-record CPU sandbox pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.pilot import records, simulator_source_hash  # noqa: E402
from src.data.validation import validate_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/pilot/v3")
    args = parser.parse_args()
    if args.full:
        parser.error("Final dataset generation is gated on ScienceWorld validation; use --smoke")
    if args.dry_run:
        print(
            "Plan: 24 sandbox pilot records, 12 matched pairs, 4 shared template groups; no models or final splits."
        )
        return
    rows = records(ROOT)
    report = validate_records(rows, expected_source_hash=simulator_source_hash(ROOT))
    if report["status"] != "PASS":
        raise SystemExit(json.dumps(report, indent=2))
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    report["dataset_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("sandbox_pairs.jsonl", payload),
        ("validation.json", json.dumps(report, indent=2) + "\n"),
    ):
        target = args.output_dir / name
        if target.exists() and target.read_text() != text:
            raise SystemExit(
                f"Refusing to overwrite changed output: {target}; select a new version directory"
            )
        target.write_text(text)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
