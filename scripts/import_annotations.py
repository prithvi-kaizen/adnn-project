"""Require completed forms and named human independence attestations."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.annotation import agreement_report, blinded_package, completed_labels  # noqa: E402
from src.data.splits import components  # noqa: E402
from src.data.validation import validate_records  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--human-attestations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.package / "manifest.json").read_text())
    if (manifest.get("scope") != "final_frozen_dual_environment_annotation" or
            manifest.get("final_sample_gate_passed") is not True or manifest.get("n") != 100):
        raise SystemExit("Only the frozen final 100-case dual-environment package can supply final agreement")
    payload = args.dataset.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["dataset_sha256"]:
        raise SystemExit("Dataset hash differs from the issued sample")
    dataset = [json.loads(line) for line in payload.splitlines()]
    if validate_records(dataset, require_final=True)["status"] != "PASS":
        raise SystemExit("Final dataset certification must pass before importing human judgments")
    grouping = components(dataset)
    if manifest.get("seed") != 4242:
        raise SystemExit("Final annotation sample must use the frozen seed")
    expected_package = blinded_package(dataset, n=100, seed=4242)
    for name in ("cases_A.csv", "cases_B.csv", "private_key.csv"):
        source = args.package / ("private/private_key.csv" if name == "private_key.csv" else name)
        if source.read_text() != expected_package[name]:
            raise SystemExit("Issued cases/key differ from the frozen seeded source sample")
    attestations = json.loads(args.human_attestations.read_text())
    if len(attestations) != 2 or len({a["human_name"] for a in attestations}) != 2 or any(
        not a["human_name"].strip() or not a["completed_at"] or
        a.get("independent_without_automatic_labels") is not True for a in attestations
    ):
        raise SystemExit("Two distinct named humans must attest independent completed annotation")
    key = list(csv.DictReader((args.package / "private/private_key.csv").open()))
    ids = {r["annotation_id"] for r in key}
    a_text, b_text = args.annotator_a.read_text(), args.annotator_b.read_text()
    a, b = completed_labels(a_text, ids), completed_labels(b_text, ids)
    auto = {r["annotation_id"]: r["automatic_label"] for r in key}
    if len(key) != 100 or len(ids) != 100 or any(r["source_id"] not in grouping for r in key):
        raise SystemExit("Final private key needs 100 unique known dataset IDs")
    groups = {r["annotation_id"]: grouping[r["source_id"]] for r in key}
    report = agreement_report(a, b, auto, groups)
    report["source_sha256"] = {"A": hashlib.sha256(a_text.encode()).hexdigest(),
                                "B": hashlib.sha256(b_text.encode()).hexdigest(),
                                "dataset": hashlib.sha256(payload).hexdigest(),
                                "attestations": hashlib.sha256(args.human_attestations.read_bytes()).hexdigest()}
    if args.output.exists():
        raise SystemExit("Preserve the prior agreement report; choose a versioned output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Completed human forms validated; disagreement originals preserved.")


if __name__ == "__main__":
    main()
