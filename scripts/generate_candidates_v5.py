"""Preserve v4 and append audited multi-object candidate families."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.candidates_v5 import records  # noqa: E402
from src.data.validation import validate_records  # noqa: E402


def main():
    out = ROOT / "data/candidates/v5"
    rows = records(ROOT)
    report = validate_records(rows)
    if report["status"] != "PASS":
        raise SystemExit(json.dumps(report, indent=2))
    payload = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    report.update(dataset_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                  scope="sandbox_candidates_not_final_dual_environment_data")
    out.mkdir(parents=True, exist_ok=True)
    for name, text in (("sandbox_pairs.jsonl", payload), ("validation.json", json.dumps(report, indent=2) + "\n")):
        path = out / name
        if path.exists() and path.read_text() != text:
            raise SystemExit("Refuse to overwrite a different issued version")
        path.write_text(text)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
