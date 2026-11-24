"""Build blinded cloud handoffs and score returned model labels against sandbox replay."""

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.annotation import LABELS  # noqa: E402
from src.sandbox.core import Action, State, ToolError, transition  # noqa: E402
from src.sandbox.labeling import label_action, paid_recovery_cost  # noqa: E402

POOL = Path("data/course_fast_track/v2")
CODEBOOK = Path("docs/LLM_ANNOTATOR_CODEBOOK.md")
CONFIG = Path("configs/course_llm_annotation_v1.json")
AUDITORS = ("A", "B")
CATEGORIES = (*LABELS, "unknown")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(source):
    def unique(items):
        mapping = dict(items)
        if len(mapping) != len(items):
            raise ValueError("Duplicate JSON object key")
        return mapping

    return json.loads(source.read_text(), object_pairs_hook=unique)


def audit_input():
    config = load_json(ROOT / CONFIG)
    plan = load_json(ROOT / "configs/course_improvement_v1.json")
    freeze = ROOT / POOL / "freeze_manifest.json"
    assert sha(freeze) == plan["protected_baseline_sha256"][str(POOL / freeze.name)]
    files = load_json(freeze)["files"]
    dataset = ROOT / POOL / "sandbox_pool.jsonl"
    split = ROOT / POOL / "splits/pool_1800/seed_1729.json"
    for path in (dataset, split):
        assert sha(path) == files[str(path.relative_to(ROOT / POOL))]["sha256"], path
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert len(rows) == 1800 and all(row["environment"] == "sandbox" for row in rows)
    assert config["sample_n"] == 100 and config["batch_size"] == 20
    sampled = random.Random(config["seed"]).sample(sorted(rows, key=lambda row: row["id"]), config["sample_n"])
    key = {f"case_{index:03}": row for index, row in enumerate(sampled, 1)}
    groups = load_json(split)["components"]
    return config, key, {case: groups[row["id"]] for case, row in key.items()}


def batches_for(auditor, config, key):
    ids = list(key)
    random.Random(config["review_order_seeds"][auditor]).shuffle(ids)
    size = config["batch_size"]
    return [ids[start:start + size] for start in range(0, len(ids), size)]


def pack_case(case, row):
    return {"case_id": case, "horizon": row["horizon"],
            "before": json.loads(row["state_text"]), "candidate": json.loads(row["action_text"])}


def prompt(auditor, batch, ids, key):
    cases = [pack_case(case, key[case]) for case in ids]
    return (f"# Audit {auditor}, batch {batch:02}\n\n"
            + (ROOT / CODEBOOK).read_text()
            + "\n## Return format\n\nReturn one JSON array with exactly one record per supplied case. "
            "Use exactly these keys: `case_id`, `label`, `reason`, `witness`. "
            "Keep your reason to one short sentence. The witness is a JSON list of legal tool objects "
            "for reversible or partially_reversible, and null otherwise. Do not wrap the array in prose "
            "or Markdown fences. Do not invent missing cases.\n\n"
            + "```json\n" + json.dumps(cases, ensure_ascii=False, separators=(",", ":")) + "\n```\n")


def oracle_replay(key):
    counts = Counter()
    for row in key.values():
        state = State.from_object(json.loads(row["state_text"]))
        action = Action.parse(row["action_text"])
        assert state.digest() == row["state_hash"]
        for horizon in (1, 2, 3):
            replayed = label_action(state, action, horizon).object()
            expected = row["labels_by_horizon"][str(horizon)]
            assert replayed == expected and replayed["label"] in LABELS, row["id"]
        assert row["label"] == row["labels_by_horizon"][str(row["horizon"])]["label"]
        counts[row["label"]] += 1
    return dict(counts)


def prepare(destination, receipt):
    config, key, groups = audit_input()
    classes = oracle_replay(key)
    orders = {auditor: batches_for(auditor, config, key) for auditor in AUDITORS}
    assert orders["A"] != orders["B"]
    contents = {auditor: {f"batch_{number:02}.md": prompt(auditor, number, ids, key)
                          for number, ids in enumerate(order, 1)}
                for auditor, order in orders.items()}
    destination.mkdir(parents=True, exist_ok=False)
    for auditor, items in contents.items():
        path = destination / f"blind_audit_{auditor}.zip"
        with ZipFile(path, "x") as bundle:
            for name, content in items.items():
                bundle.writestr(ZipInfo(name), content.encode(), compress_type=ZIP_DEFLATED)
        with ZipFile(path) as bundle:
            assert bundle.testzip() is None and bundle.namelist() == list(items)
            assert all(bundle.read(name).decode() == content for name, content in items.items())
    inputs = [CONFIG, CODEBOOK, POOL / "freeze_manifest.json", POOL / "sandbox_pool.jsonl",
              POOL / "splits/pool_1800/seed_1729.json", Path(__file__).resolve().relative_to(ROOT)]
    report = {"status": "PACKETS_VERIFIED_AWAITING_MODEL_RETURNS", "scope": config["scope"],
              "unique_cases": len(key), "batches_per_model": len(orders["A"]), "cases_per_batch": config["batch_size"],
              "eligible_pool": 1800, "sample_components": len(set(groups.values())),
              "oracle_horizon_replays": 3 * len(key), "private_sample_class_counts": classes,
              "source_sha256": {str(path): sha(ROOT / path) for path in inputs},
              "packet_sha256": {path.name: sha(path) for path in sorted(destination.glob("*.zip"))},
              "verified_fields_per_case": ["case_id", "horizon", "before", "candidate"],
              "automatic_labels_in_packets": False, "new_cloud_calls_made_by_project": 0,
              "model_returns_received": 0, "human_annotations": 0, "final_joint_sample": False}
    save(receipt, report)
    return report


def validate_batch(records, expected, horizon):
    if type(records) is not list:
        raise ValueError("Return must be a JSON array")
    found = {}
    for record in records:
        if type(record) is not dict or set(record) != {"case_id", "label", "reason", "witness"}:
            raise ValueError("Each record needs exactly case_id, label, reason and witness")
        case, label, reason, witness = (record[k] for k in ("case_id", "label", "reason", "witness"))
        if type(case) is not str or case in found or case not in expected:
            raise ValueError("Unexpected or duplicate case ID")
        if label not in CATEGORIES or type(reason) is not str or not reason.strip():
            raise ValueError(f"Invalid label or empty reason: {case}")
        if label in LABELS[:2]:
            if type(witness) is not list or len(witness) > horizon[case]:
                raise ValueError(f"Expected a list no longer than the horizon: {case}")
            for action in witness:
                Action.parse(json.dumps(action))
        elif witness is not None:
            raise ValueError(f"Expected null witness: {case}")
        found[case] = record
    if set(found) != set(expected):
        raise ValueError(f"Missing cases: {sorted(set(expected) - set(found))}")
    return found


def witness_result(record, row):
    if record["witness"] is None:
        return "not_claimed"
    state = State.from_object(json.loads(row["state_text"]))
    current, _ = transition(state, Action.parse(row["action_text"]))
    try:
        for action in record["witness"]:
            current, _ = transition(current, Action.parse(json.dumps(action)))
    except ToolError:
        return "illegal_recovery_action"
    if current.digest() == state.digest():
        return "exact"
    if paid_recovery_cost(state, current) is not None:
        return "paid"
    return "not_restored"


def compare(left, right, groups):
    ids = sorted(left)
    assert ids and set(ids) == set(right) == set(groups)
    clustered = {}
    for case in ids:
        clustered.setdefault(groups[case], []).append(case)

    def score(sample):
        left_counts, right_counts = Counter(left[case] for case in sample), Counter(right[case] for case in sample)
        length = len(sample)
        exact = sum(left[case] == right[case] for case in sample)
        chance = sum(left_counts[label] * right_counts[label] for label in CATEGORIES)
        return exact / length, ((exact * length - chance) / (length ** 2 - chance)
                                if length ** 2 > chance else None)

    estimate, kappa = score(ids)
    rng = random.Random(9001)
    names = sorted(clustered)
    draws = [score([case for name in rng.choices(names, k=len(names)) for case in clustered[name]])
             for _ in range(1000)]

    def limits(values, point):
        defined = sorted(value for value in values if value is not None)
        return {"CI95": [defined[math.ceil(len(defined) * quantile) - 1] for quantile in (.025, .975)]
                if point is not None and len(names) >= 3 and len(defined) >= 950 else None,
                "defined_draws": len(defined), "undefined_draws": 1000 - len(defined)}

    return {"N": len(ids), "clusters": len(names),
            "categories": CATEGORIES,
            "unknown_is_explicit_category": True,
            "raw_agreement": {"estimate": estimate, **limits([draw[0] for draw in draws], estimate)},
            "cohens_kappa": {"estimate": kappa, **limits([draw[1] for draw in draws], kappa)},
            "confusion_rows_left_columns_right": [
                [sum(left[case] == first and right[case] == second for case in ids) for second in CATEGORIES]
                for first in CATEGORIES]}


def review_returns(directory, packet_directory, prep_receipt):
    config, key, groups = audit_input()
    prepared = load_json(prep_receipt)
    assert prepared["status"] == "PACKETS_VERIFIED_AWAITING_MODEL_RETURNS"
    assert all(sha(ROOT / source) == expected for source, expected in prepared["source_sha256"].items())
    assert all(sha(packet_directory / name) == expected for name, expected in prepared["packet_sha256"].items())
    meta = load_json(directory / "metadata.json")
    if type(meta) is not dict or not meta or not set(meta) <= set(AUDITORS):
        raise ValueError("Give metadata for each submitted reviewer, A and/or B")
    auditors = [auditor for auditor in AUDITORS if auditor in meta]
    for auditor in auditors:
        item = meta[auditor]
        fields = ("provider", "model_as_displayed", "declared_family", "settings", "run_date")
        if type(item) is not dict or set(item) != {*fields, "blinded_separate_sessions"}:
            raise ValueError(f"Unexpected metadata fields for {auditor}")
        for field in fields:
            if type(item.get(field)) is not str or not item[field].strip():
                raise ValueError(f"Missing {auditor} metadata: {field}")
        date.fromisoformat(item["run_date"])
        if item.get("blinded_separate_sessions") is not True:
            raise ValueError("The user must attest to separate sessions with no labels or other outputs")
    replay = oracle_replay(key)
    oracle = {case: row["label"] for case, row in key.items()}
    horizon = {case: row["horizon"] for case, row in key.items()}
    normalized, witnesses = {}, {}
    hashes = {"metadata.json": sha(directory / "metadata.json")}
    for auditor in auditors:
        found = {}
        for batch, ids in enumerate(batches_for(auditor, config, key), 1):
            name = f"{auditor}/{batch:02}.json"
            path = directory / name
            hashes[name] = sha(path)
            found.update(validate_batch(load_json(path), ids, horizon))
        normalized[auditor] = found
        witnesses[auditor] = {case: witness_result(record, key[case]) for case, record in found.items()}
    labels = {auditor: {case: record["label"] for case, record in normalized[auditor].items()}
              for auditor in auditors}
    model_reports = {}
    for auditor, predictions in labels.items():
        report = compare(predictions, oracle, groups)
        correct = sum(predictions[case] == oracle[case] for case in key)
        recall = {label: sum(predictions[case] == oracle[case] == label for case in key) / replay[label]
                  if replay.get(label) else None for label in LABELS}
        false_recoverable = sum(oracle[case] == "irreversible" and predictions[case] in LABELS[:2] for case in key)
        invalid_witnesses = sum(predictions[case] in LABELS[:2] and
                                witnesses[auditor][case] != ("exact" if predictions[case] == LABELS[0] else "paid")
                                for case in key)
        rule = config["pilot_quality_rule"]
        report.update(correct=correct, abstentions=sum(value == "unknown" for value in predictions.values()),
                      class_recall=recall, false_recoverable=false_recoverable, invalid_witnesses=invalid_witnesses,
                      passes_quality_rule=(correct >= rule["min_correct_of_100"] and
                                           false_recoverable <= rule["max_false_recoverable"] and
                                           invalid_witnesses <= rule["max_invalid_witnesses"] and
                                           all(value is not None and
                                               value >= rule["min_class_recall"] for value in recall.values())))
        model_reports[auditor] = report
    complete = len(auditors) == len(AUDITORS)
    distinct = complete and len({item["declared_family"].strip().casefold() for item in meta.values()}) == 2
    flagged = []
    for case in sorted(key):
        responses = {auditor: {**normalized[auditor][case], "witness_result": witnesses[auditor][case]}
                     for auditor in auditors}
        if (len({oracle[case], *(labels[auditor][case] for auditor in auditors)}) > 1 or
                any(item["label"] in LABELS[:2] and
                    item["witness_result"] != ("exact" if item["label"] == LABELS[0] else "paid")
                    for item in responses.values())):
            flagged.append({"case_id": case, "oracle": oracle[case], "models": responses})
    return {"status": "MODEL_RETURNS_AUDITED" if complete else "SINGLE_REVIEWER_AUDITED",
            "scope": config["scope"], "human_annotations": 0, "complete_reviewer_returns": auditors,
            "bootstrap_seed": 9001, "bootstrap_draws": 1000, "external_return_sha256": hashes,
            "preparation_receipt_sha256": sha(prep_receipt), "model_metadata": meta,
            "distinct_declared_model_families": distinct,
            "model_to_model": compare(labels["A"], labels["B"], groups) if complete else None,
            "model_to_oracle": model_reports, "disagreements_or_invalid_witnesses": flagged,
            "pilot_quality_rule_met": complete and (distinct or not rule["two_distinct_declared_families_required"]) and
            all(r["passes_quality_rule"] for r in model_reports.values()),
            "environments": ["sandbox"], "final_joint_sample": False, "human_equivalence_established": False,
            "relabelled_examples": 0, "original_gate_2_complete": False}


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--packets", type=Path, required=True)
    build.add_argument("--receipt", type=Path, required=True)
    review = commands.add_parser("review")
    review.add_argument("--packets", type=Path, required=True)
    review.add_argument("--receipt", type=Path, required=True)
    review.add_argument("--returns", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.receipt.exists():
            parser.error("Choose a new preparation receipt")
        report = prepare(args.packets, args.receipt)
    else:
        if args.output.exists():
            parser.error("Choose a new audit report")
        report = review_returns(args.returns, args.packets, args.receipt)
        save(args.output, report)
    print(report["status"])


if __name__ == "__main__":
    main()
