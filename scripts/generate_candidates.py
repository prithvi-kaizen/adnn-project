"""Generate validated candidates without clearing the final dual-environment gate."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.candidates import records  # noqa: E402
from src.data.validation import validate_records  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-per-mechanism", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/candidates/v4")
    args = parser.parse_args()
    rows = records(ROOT, pairs_per_mechanism=args.pairs_per_mechanism)
    report = validate_records(rows)
    if report["status"] != "PASS":
        raise SystemExit(json.dumps(report, indent=2))
    payload = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    report.update(dataset_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                  scope="sandbox_candidates_not_final_dual_environment_data")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in [("sandbox_pairs.jsonl", payload),
                          ("validation.json", json.dumps(report, indent=2) + "\n")]:
        p = args.output_dir / name
        if p.exists() and p.read_text() != content:
            raise SystemExit(f"Refuse changed overwrite: {p}")
        p.write_text(content)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
