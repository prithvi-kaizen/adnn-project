"""Connected-component grouping and explicit candidate split diagnostics."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict


def structural_state_signature(state_text: str) -> str:
    """Conservative near-duplicate key, preserving value equality and topology.

    Names/content are consistently anonymized; quantities retain zero/positive
    status. This key only joins groups, never certifies state equivalence/labels.
    """
    strings = {}

    def abstract(value):
        if type(value) is str:
            if value not in strings:
                strings[value] = f"string_{len(strings)}"
            return strings[value]
        if type(value) is bool or value is None:
            return value
        if type(value) in (int, float):
            return "zero" if value == 0 else "positive" if value > 0 else "negative"
        if type(value) is list:
            return [abstract(x) for x in value]
        if type(value) is dict:
            return [(abstract(k), abstract(v)) for k, v in value.items()]
        raise ValueError("Unsupported JSON value")

    # Preserve schema field names, anonymize symbolic keys and contents within fields.
    obj = json.loads(state_text)
    normalized = [(key, abstract(obj[key])) for key in sorted(obj)]
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def components(rows: list[dict]) -> dict[str, str]:
    parent = {row["id"]: row["id"] for row in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    seen = {}
    for row in rows:
        keys = [(name, row.get(name)) for name in
                ("scenario_id", "template_id", "minimal_pair_id", "split_group_id") if row.get(name)]
        keys.append(("state", row["environment"], row["state_hash"]))
        if row["environment"] == "sandbox":
            keys.append(("near_state", structural_state_signature(row["state_text"])))
        for key in keys:
            if key in seen:
                union(row["id"], seen[key])
            else:
                seen[key] = row["id"]
    return {key: find(key) for key in parent}


def candidate_split(rows: list[dict], seed: int) -> dict:
    """Find closest group-respecting 60/20/20 split, disclosing any deviation.

    Intended for small candidate pools, not a substitute for the final freeze.
    No model outputs or scores enter the allocation.
    """
    groups = components(rows)
    members = defaultdict(list)
    for row in rows:
        members[groups[row["id"]]].append(row)
    names = sorted(members)
    random.Random(seed).shuffle(names)
    if len(names) > 12:
        return integer_candidate_split(rows, seed, groups, members, names)
    classes = {r["label"] for r in rows}
    n = len(rows)
    best = None
    for assignment in itertools.product(("train", "development", "test"), repeat=len(names)):
        counts = Counter()
        coverage = defaultdict(set)
        for name, split in zip(names, assignment):
            counts[split] += len(members[name])
            coverage[split].update(r["label"] for r in members[name])
        if any(coverage[split] != classes for split in ("train", "development", "test")):
            continue
        error = sum(abs(counts[s] / n - fraction) for s, fraction in
                    (("train", .6), ("development", .2), ("test", .2)))
        if best is None or error < best[0]:
            best = error, assignment, dict(counts)
    if best is None:
        return {"status": "BLOCKED", "reason": "insufficient_independent_class_coverage",
                "component_count": len(names), "seed": seed}
    _, assignment, counts = best
    allocation = dict(zip(names, assignment))
    return {"status": "CANDIDATE_ONLY", "seed": seed, "component_count": len(names),
            "counts": counts, "fractions": {s: counts[s] / n for s in counts},
            "exact_60_20_20": all(counts[s] * 5 == n * k for s, k in
                                    (("train", 3), ("development", 1), ("test", 1))),
            "final_freeze": False,
            "indices": {s: [r["id"] for r in rows if allocation[groups[r["id"]]] == s]
                        for s in ("train", "development", "test")},
            "components": groups}


def validate_split(rows: list[dict], manifest: dict) -> None:
    groups = components(rows)
    assigned = {}
    for split, ids in manifest["indices"].items():
        if split not in {"train", "development", "test"}:
            raise ValueError("Invalid split")
        for row_id in ids:
            if row_id in assigned or row_id not in groups:
                raise ValueError("Duplicate or unknown split ID")
            assigned[row_id] = split
    if set(assigned) != set(groups):
        raise ValueError("Missing split IDs")
    grouped = defaultdict(set)
    for row_id, split in assigned.items():
        grouped[groups[row_id]].add(split)
    if any(len(splits) > 1 for splits in grouped.values()):
        raise ValueError("Connected component leaks across splits")


def integer_candidate_split(rows, seed, groups, members, names):
    """Assign whole components with exact count deviations and class coverage.

    Uses the pinned research environment's SciPy/HiGHS, not a model-dependent
    heuristic. The small-pool exhaustive planner remains dependency-free.
    """
    import numpy as np
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp

    splits = ("train", "development", "test")
    count = len(names)
    width = 3 * count + 3
    constraints, lower, upper = [], [], []

    def add(coefficients, low, high):
        constraints.append(coefficients)
        lower.append(low)
        upper.append(high)

    for i in range(count):
        coefficients = np.zeros(width)
        coefficients[3 * i:3 * i + 3] = 1
        add(coefficients, 1, 1)
    for j in range(3):
        for label in sorted({r["label"] for r in rows}):
            coefficients = np.zeros(width)
            for i, name in enumerate(names):
                coefficients[3 * i + j] = any(r["label"] == label for r in members[name])
            add(coefficients, 1, np.inf)
        size = np.zeros(width)
        for i, name in enumerate(names):
            size[3 * i + j] = len(members[name])
        target = len(rows) * (0.6 if j == 0 else 0.2)
        for sign in (1, -1):
            coefficients = sign * size
            coefficients[3 * count + j] = -1
            add(coefficients, -np.inf, sign * target)
    # Count deviations dominate; bounded seed-dependent tie costs never exceed .01 records.
    objective = np.r_[np.arange(1, 3 * count + 1) * (.01 / (3 * count) ** 2), np.ones(3)]
    result = milp(objective, integrality=np.r_[np.ones(3 * count), np.zeros(3)],
                  bounds=Bounds(np.zeros(width), np.r_[np.ones(3 * count), np.full(3, np.inf)]),
                  constraints=LinearConstraint(np.array(constraints), lower, upper),
                  options={"time_limit": 30, "mip_rel_gap": 0})
    if result.status != 0:
        return {"status": "BLOCKED", "reason": "integer_allocator_not_optimal",
                "solver_message": str(result.message), "seed": seed, "component_count": count}
    assignment = result.x[:3 * count].reshape(count, 3).argmax(axis=1)
    allocation = {name: splits[int(j)] for name, j in zip(names, assignment)}
    indices = {s: [r["id"] for r in rows if allocation[groups[r["id"]]] == s] for s in splits}
    counts = {s: len(ids) for s, ids in indices.items()}
    manifest = {"status": "CANDIDATE_ONLY", "seed": seed, "component_count": count,
                "counts": counts, "fractions": {s: counts[s] / len(rows) for s in splits},
                "exact_60_20_20": all(counts[s] * 5 == len(rows) * k for s, k in zip(splits, (3, 1, 1))),
                "final_freeze": False, "indices": indices, "components": groups,
                "allocator": "scipy.optimize.milp_HiGHS", "scipy_version": scipy.__version__,
                "objective": "minimum_absolute_record_count_deviation_with_class_coverage"}
    validate_split(rows, manifest)
    return manifest
