"""Derive candidate counts, horizon shifts and identifier checks from JSONL."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.splits import components  # noqa: E402
from src.data.validation import validate_records  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.dataset.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    validation = validate_records(rows)
    report = {"rows": len(rows), "mechanisms": {}, "by_horizon": {}, "identifier_leaks": [],
              "source_dataset": str(args.dataset), "dataset_sha256": hashlib.sha256(payload).hexdigest(),
              "sandbox_validation_status": validation["status"],
              "connected_components": len(set(components(rows).values()))}
    for mechanism in sorted({r["mechanism"] for r in rows}):
        report["mechanisms"][mechanism] = dict(Counter(r["label"] for r in rows if r["mechanism"] == mechanism))
    for h in ("1", "2", "3"):
        report["by_horizon"][h] = dict(Counter(r["labels_by_horizon"][h]["label"] for r in rows))
    for row in rows:
        rendered = row["state_text"] + row["action_text"] + row["goal"]
        if any(row[k] in rendered for k in ("id", "scenario_id", "template_id", "minimal_pair_id") if row.get(k)):
            report["identifier_leaks"].append(row["id"])
    report["horizon_changed_ids"] = [r["id"] for r in rows if len({
        r["labels_by_horizon"][h]["label"] for h in ("1", "2", "3")}) > 1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if validation["status"] != "PASS" or report["identifier_leaks"]:
        raise SystemExit("Candidate quality check failed")
    print(json.dumps({k: v for k, v in report.items() if k not in {"mechanisms", "horizon_changed_ids"}}, indent=2))


if __name__ == "__main__":
    main()
