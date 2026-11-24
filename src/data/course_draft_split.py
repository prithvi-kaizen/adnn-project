"""Feasibility-only sandbox partitions with conservative exposure constraints."""

import random
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from src.probes.course import eligible_indices
from src.probes.linear import LABEL_ORDER

SPLITS = ("train", "development", "test")
SEEDS = (1729, 2718, 31415)


def coverage_catalog(rows):
    catalog = {}
    for horizon in (1, 2, 3):
        for binary in (False, True):
            options = ("reversible", "irreversible") if binary else LABEL_ORDER
            for paired in (False, True):
                eligible = eligible_indices(rows, horizon, binary, paired)
                name = f"h{horizon}_{'binary' if binary else 'threeway'}_{'pairs' if paired else 'all'}"
                catalog[name] = {label: {rows[index]["id"] for index in eligible
                                         if rows[index]["labels_by_horizon"][str(horizon)]["label"] == label}
                                 for label in options}
    return catalog


def allocate_draft_splits(rows, assignments):
    """Find exact 20% test splits without evaluating or authorizing a holdout."""
    by_group = defaultdict(set)
    for row in rows:
        by_group[assignments[row["id"]]["group"]].add(row["id"])
    names = sorted(by_group)
    count = len(names)
    if len(rows) % 5 or any(len({assignments[key]["quarantined"] for key in group}) != 1
                            for group in by_group.values()):
        raise ValueError("Cannot request exact fifths or use mixed-status components")
    exposed = [any(assignments[key]["quarantined"] for key in by_group[name]) for name in names]
    target = len(rows) // 5
    catalog = coverage_catalog(rows)
    result = []
    past_test = []
    for seed in SEEDS:
        matrix, lower, upper = [], [], []
        dimension = 3 * count + 1

        def constrain(coefficients, minimum, maximum):
            matrix.append(coefficients)
            lower.append(minimum)
            upper.append(maximum)

        for index in range(count):
            vector = np.zeros(dimension)
            vector[3 * index:3 * index + 3] = 1
            constrain(vector, 1, 1)
        for partition in range(3):
            columns = slice(partition, 3 * count, 3)
            size = np.zeros(dimension)
            size[columns] = [len(by_group[name]) for name in names]
            if partition == 2:
                constrain(size, target, target)
                minimum_groups = np.zeros(dimension)
                minimum_groups[columns] = 1
                constrain(minimum_groups, 8, np.inf)
                for earlier in past_test:
                    differ = np.zeros(dimension)
                    differ[columns] = [name in earlier for name in names]
                    constrain(differ, -np.inf, len(earlier) - 1)
            if partition == 1:
                for sign in (-1, 1):
                    deviation = sign * size
                    deviation[-1] = -1
                    constrain(deviation, -np.inf, sign * target)
            for coverage in catalog.values():
                for ids in coverage.values():
                    vector = np.zeros(dimension)
                    vector[columns] = [bool(by_group[name] & ids) for name in names]
                    constrain(vector, 1, np.inf)
        cap = np.ones(dimension)
        cap[2:3 * count:3] = [not value for value in exposed]
        cap[-1] = np.inf
        rng = random.Random(seed)
        objective = [rng.random() / (300 * count) for _ in range(3 * count)] + [1.]
        solved = milp(objective, integrality=np.r_[np.ones(3 * count), 0],
                      bounds=Bounds(np.zeros(dimension), cap),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={"mip_rel_gap": 0, "time_limit": 30})
        if solved.status != 0:
            raise ValueError(f"No certified draft allocation for seed {seed}: {solved.message}")
        one_hot = solved.x[:3 * count].reshape(count, 3)
        assert np.allclose(one_hot, np.rint(one_hot)) and np.allclose(one_hot.sum(1), 1)
        allocation = {name: SPLITS[index] for name, index in zip(names, one_hot.argmax(1), strict=True)}
        test_groups = {name for name in names if allocation[name] == "test"}
        assert test_groups not in past_test and len(test_groups) >= 8
        past_test.append(test_groups)
        indices = {partition: [row["id"] for row in rows
                               if allocation[assignments[row["id"]]["group"]] == partition]
                   for partition in SPLITS}
        assert len(indices["test"]) == target
        assert all(not assignments[key]["quarantined"] for key in indices["test"])
        assert Counter(key for members in indices.values() for key in members) == Counter(assignments.keys())
        support = {}
        for partition, members in indices.items():
            cohort = set(members)
            support[partition] = {}
            for name, labels in catalog.items():
                counts = {label: len(cohort & eligible) for label, eligible in labels.items()}
                assert min(counts.values()) >= 1
                support[partition][name] = counts
        result.append({"status": "DRAFT_FEASIBILITY_ONLY", "seed": seed, "indices": indices,
                       "group_counts": dict(Counter(allocation.values())), "test_groups": sorted(test_groups),
                       "class_counts": support, "reuse_allowed_for_model_runs": False,
                       "used_model_outcomes": False, "exact_20_percent_test": True,
                       "minimum_test_group_count": 8, "test_groups_linked_to_history": 0})
    return result
