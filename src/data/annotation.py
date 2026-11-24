"""Blinded forms and agreement calculations; never synthesize human labels."""

from __future__ import annotations

import csv
import random
from io import StringIO

LABELS = ("reversible", "partially_reversible", "irreversible")


def csv_text(rows: list[dict], fields: list[str]) -> str:
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def blinded_package(rows: list[dict], n: int = 100, seed: int = 4242) -> dict[str, str]:
    if len(rows) < n or n < 1:
        raise ValueError("Need enough distinct eligible examples")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source IDs")
    if any(r["label"] not in LABELS for r in rows):
        raise ValueError("Annotation pool contains unknown/invalid labels")
    sampled = random.Random(seed).sample(sorted(rows, key=lambda r: r["id"]), n)
    cases, forms, key = [], [], []
    for i, row in enumerate(sampled):
        blind_id = f"case_{i + 1:03d}"
        cases.append({"annotation_id": blind_id, **{k: row[k] for k in
                      ("environment", "goal", "state_text", "action_text", "horizon")}})
        forms.append({"annotation_id": blind_id, "label": "", "confidence": "", "reason": ""})
        key.append({"annotation_id": blind_id, "source_id": row["id"], "automatic_label": row["label"]})
    package = {"private_key.csv": csv_text(key, list(key[0]))}
    for offset, annotator in enumerate(("A", "B"), 1):
        order = list(range(n))
        random.Random(seed + offset).shuffle(order)
        package[f"cases_{annotator}.csv"] = csv_text([cases[i] for i in order], list(cases[0]))
        package[f"annotator_{annotator}.csv"] = csv_text([forms[i] for i in order], list(forms[0]))
    return package


def completed_labels(text: str, expected_ids: set[str]) -> dict[str, str]:
    result = {}
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames != ["annotation_id", "label", "confidence", "reason"]:
        raise ValueError("Expected exact annotation form columns")
    for row in reader:
        if set(row) != {"annotation_id", "label", "confidence", "reason"}:
            raise ValueError("Malformed annotation row")
        if row["confidence"] not in {"low", "medium", "high"} or not (row["reason"] or "").strip():
            raise ValueError("Confidence and independent rationale required")
        key, label = row["annotation_id"], row["label"]
        if key in result or key not in expected_ids or label not in LABELS:
            raise ValueError("Missing, duplicate or invalid human label")
        result[key] = label
    if set(result) != expected_ids:
        raise ValueError("Incomplete annotation IDs")
    return result


def cohens_kappa(left: list[str], right: list[str]) -> dict:
    if not left or len(left) != len(right) or any(x not in LABELS for x in left + right):
        raise ValueError("Expected aligned nonempty legal labels")
    n = len(left)
    agreement = sum(a == b for a, b in zip(left, right)) / n
    chance = sum(left.count(label) * right.count(label) for label in LABELS) / n ** 2
    return {"n": n, "observed_agreement": agreement, "expected_agreement": chance,
            "cohens_kappa": (agreement - chance) / (1 - chance) if chance < 1 else None,
            "undefined_reason": "constant_identical_class" if chance == 1 else None}


def agreement_report(a: dict[str, str], b: dict[str, str], automatic: dict[str, str],
                     groups: dict[str, str], *, resamples: int = 1000, seed: int = 9001) -> dict:
    """Compare genuine completed inputs; tests must identify synthetic fixtures."""
    if not a or set(a) != set(b) or set(a) != set(automatic) or set(a) != set(groups):
        raise ValueError("Agreement requires complete aligned IDs and cluster provenance")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("Positive bootstrap count required")
    clustered = {}
    for key in sorted(a):
        clustered.setdefault(groups[key], []).append(key)
    if any(type(g) is not str or not g for g in clustered):
        raise ValueError("Every annotation requires a nonempty group")
    ids, names = sorted(a), sorted(clustered)
    rng = random.Random(seed)
    draws = [[key for group in rng.choices(names, k=len(names)) for key in clustered[group]]
             for _ in range(resamples)]

    def compare(left, right):
        base = cohens_kappa([left[k] for k in ids], [right[k] for k in ids])
        samples, undefined = [], 0
        for sample in draws:
            value = cohens_kappa([left[k] for k in sample], [right[k] for k in sample])["cohens_kappa"]
            if value is None:
                undefined += 1
            else:
                samples.append(value)
        samples.sort()
        # Nearest-rank percentile intervals; disclose undefined cluster resamples.
        import math
        ci = ([samples[max(0, math.ceil(.025 * len(samples)) - 1)],
               samples[max(0, math.ceil(.975 * len(samples)) - 1)]]
              if base["cohens_kappa"] is not None and len(samples) >= .95 * resamples else None)
        return {**base, "cluster_bootstrap_95_ci": ci, "bootstrap_valid": len(samples),
                "bootstrap_undefined": undefined,
                "ci_undefined_reason": None if ci else "undefined_estimate_or_over_5_percent_degenerate_resamples",
                "confusion_rows_left_columns_right": [[sum(left[k] == x and right[k] == y for k in ids)
                                                        for y in LABELS] for x in LABELS],
                "left_class_counts": {label: sum(left[k] == label for k in ids) for label in LABELS},
                "right_class_counts": {label: sum(right[k] == label for k in ids) for label in LABELS}}
    return {"n": len(ids), "cluster_count": len(names), "bootstrap_seed": seed,
            "bootstrap_resamples": resamples, "label_order": LABELS,
            "human_human": compare(a, b), "human_A_auto": compare(a, automatic),
            "human_B_auto": compare(b, automatic),
            "disagreements": [{"annotation_id": k, "A": a[k], "B": b[k], "auto": automatic[k],
                               "human_human_disagrees": a[k] != b[k],
                               "human_A_auto_disagrees": a[k] != automatic[k],
                               "human_B_auto_disagrees": b[k] != automatic[k]}
                              for k in ids if len({a[k], b[k], automatic[k]}) > 1],
            "adjudication_status": "PENDING_SEPARATE_HUMAN_RECORD",
            "missing_annotations": {"A": 0, "B": 0},
            "few_clusters_warning": len(names) < 20}


def validate_blank_package(package: dict[str, str], expected_n: int = 100) -> dict:
    """Check identical blinded content, independent ordering and empty judgment fields."""
    orders, contents = {}, {}
    for person in ('A', 'B'):
        cases = list(csv.DictReader(StringIO(package[f'cases_{person}.csv'])))
        forms = list(csv.DictReader(StringIO(package[f'annotator_{person}.csv'])))
        ids = [r['annotation_id'] for r in cases]
        if (len(ids) != expected_n or len(set(ids)) != expected_n or
                [r['annotation_id'] for r in forms] != ids):
            raise ValueError('Cases and blank forms must align with distinct IDs')
        if any(set(r) != {'annotation_id', 'label', 'confidence', 'reason'} or
               any(r[k] != '' for k in ('label', 'confidence', 'reason')) for r in forms):
            raise ValueError('Issued forms must contain no judgments')
        if any(set(r) != {'annotation_id', 'environment', 'goal', 'state_text', 'action_text', 'horizon'}
               for r in cases):
            raise ValueError('Blinded cases have unexpected or missing fields')
        orders[person], contents[person] = ids, {r['annotation_id']: r for r in cases}
    if contents['A'] != contents['B'] or orders['A'] == orders['B']:
        raise ValueError('Annotators need identical content and independently shuffled presentation')
    return {'status': 'PASS', 'n': expected_n, 'same_content_by_id': True,
            'different_presentation_orders': True, 'all_judgments_blank': True,
            'real_annotations_received': 0, 'final_sample_certified': False}
