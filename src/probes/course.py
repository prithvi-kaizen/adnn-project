"""Frozen course condition membership and independently checkable ID joins."""

import json
from collections import defaultdict

import numpy as np

from src.probes.linear import LABEL_ORDER


def labels_for(rows, horizon):
    return np.array([r["labels_by_horizon"][str(horizon)]["label"] for r in rows])


def eligible_indices(rows, horizon, binary, minimal_only=False):
    labels = labels_for(rows, horizon)
    allowed = ("reversible", "irreversible") if binary else LABEL_ORDER
    eligible = {i for i, label in enumerate(labels) if label in allowed}
    if minimal_only:
        pairs = defaultdict(list)
        for i, row in enumerate(rows):
            if row.get("minimal_pair_id"):
                pairs[row["minimal_pair_id"]].append(i)
        eligible = set()
        for siblings in pairs.values():
            if (len(siblings) == 2 and all(labels[i] in allowed for i in siblings) and
                    len({labels[i] for i in siblings}) == 2):
                eligible.update(siblings)
    return np.array(sorted(eligible), dtype=int)


def verb_labels(rows, label_order):
    verbs = [json.loads(row["action_text"])["tool"] for row in rows]
    generator = np.random.default_rng(777)
    mapping = {verb: str(generator.choice(label_order)) for verb in sorted(set(verbs))}
    return np.array([mapping[verb] for verb in verbs]), mapping


def conditions():
    result = []
    for seed in (1729, 2718, 31415):
        for horizon in (3, 1, 2):
            for binary in (True, False):
                for pooling in ("last", "mean"):
                    result.append(dict(seed=seed, horizon=horizon, binary=binary, pooling=pooling,
                                       subset="all", control="ordinary"))
        for binary in (True, False):
            for pooling in ("last", "mean"):
                result.append(dict(seed=seed, horizon=3, binary=binary, pooling=pooling,
                                   subset="minimal_pairs", control="ordinary"))
                for shuffle in range(101, 111):
                    result.append(dict(seed=seed, horizon=3, binary=binary, pooling=pooling,
                                       subset="all", control=f"shuffle_{shuffle}"))
                result.append(dict(seed=seed, horizon=3, binary=binary, pooling=pooling,
                                   subset="all", control="verb_777"))
    return result


def condition_id(condition):
    return "s{seed}_h{horizon}_{task}_{pooling}_{subset}_{control}".format(
        task="binary" if condition["binary"] else "threeway", **condition)


def partitions(rows, manifest, condition):
    eligible = set(eligible_indices(rows, condition["horizon"], condition["binary"],
                                    condition["subset"] == "minimal_pairs"))
    members = {key: set(ids) for key, ids in manifest["indices"].items()}
    return {key: np.array([i for i, row in enumerate(rows) if i in eligible and row["id"] in ids], dtype=int)
            for key, ids in members.items()}
