"""Exact component-bootstrap metrics from saved layer predictions."""

import numpy as np


def component_metrics(record, membership, counts, labels):
    """Compute each draw from component totals and tie-aware AUROC pair counts."""
    membership, counts = np.asarray(membership), np.asarray(counts)
    truth = np.asarray(record["truth"])
    predicted = np.asarray(record["predictions"])
    component_count = counts.shape[1]
    true_mask = truth[:, None] == np.asarray(labels)
    predicted_mask = predicted[:, None] == np.asarray(labels)
    if not true_mask.any(1).all() or not predicted_mask.any(1).all():
        raise ValueError("Unexpected class in saved layer predictions")

    def totals(mask):
        per_group = np.zeros((component_count, mask.shape[1]), dtype=np.int64)
        np.add.at(per_group, membership, mask)
        return counts @ per_group

    support = totals(true_mask)
    predicted_support = totals(predicted_mask)
    correct = totals(true_mask & predicted_mask)
    available = support > 0
    complete = available.all(1)
    total = support.sum(1)
    recall = np.divide(correct, support, out=np.zeros_like(support, dtype=float), where=available)
    denominator = support + predicted_support
    f1 = np.divide(2 * correct, denominator, out=np.zeros_like(support, dtype=float), where=denominator > 0)
    accuracy = correct.sum(1) / total
    balanced = np.where(complete, recall.mean(1), np.nan)
    auc = np.full(len(counts), np.nan)
    classes = record["classes"]
    if set(classes) == set(labels) and complete.any():
        probabilities = np.asarray(record["probabilities"])
        if (probabilities.shape != (len(truth), len(classes)) or
                not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1)):
            raise ValueError("Invalid saved probabilities")
        auc[complete] = 0
        positives = ("irreversible",) if len(labels) == 2 else labels
        for label in positives:
            positive = truth == label
            score = probabilities[:, classes.index(label)]
            pair_credit = np.zeros((component_count, component_count))
            for group in range(component_count):
                negative_scores = np.sort(score[~positive & (membership == group)])
                low = np.searchsorted(negative_scores, score[positive], side="left")
                high = np.searchsorted(negative_scores, score[positive], side="right")
                np.add.at(pair_credit[:, group], membership[positive], (low + high) * .5)
            wins = ((counts @ pair_credit) * counts).sum(1)
            pos_count = support[:, labels.index(label)]
            auc[complete] += (wins[complete] /
                              (pos_count[complete] * (total[complete] - pos_count[complete])))
        auc[complete] /= len(positives)
    return {"accuracy": accuracy, "balanced_accuracy": balanced, "macro_F1": f1.mean(1), "AUROC": auc}, complete
