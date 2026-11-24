"""Produce a clearly provisional, blinded 100-case sandbox preview."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.annotation import blinded_package  # noqa: E402
from src.data.validation import validate_records  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = args.dataset.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    validation = validate_records(rows)
    if validation["status"] != "PASS":
        raise SystemExit("Dataset must validate before sampling")
    package = blinded_package(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in package.items():
        directory = args.output_dir / "private" if name == "private_key.csv" else args.output_dir
        directory.mkdir(exist_ok=True)
        path = directory / name
        if path.exists() and path.read_text() != content:
            raise SystemExit("Refuse to change an issued annotation package")
        path.write_text(content)
    manifest = {"scope": "provisional_sandbox_preview_not_final_100_case_sample",
                "dataset_sha256": hashlib.sha256(payload).hexdigest(), "seed": 4242,
                "n": 100, "real_annotations_received": 0, "agreement_status": "PENDING",
                "final_sample_gate_passed": False,
                "instruction": "Freeze the validated dual-environment pool before issuing the final sample."
                " Do not use preview labels as the final agreement result."}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
