"""Training/development selection only; evaluation consumes saved probabilities."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABEL_ORDER = ("reversible", "partially_reversible", "irreversible")
C_GRID = (.01, .1, 1, 10)


def metrics(truth, predictions, probabilities, classes, label_order=LABEL_ORDER):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    probabilities = np.asarray(probabilities)
    if (len(truth) == 0 or truth.shape != predictions.shape or
            probabilities.shape != (len(truth), len(classes)) or
            not np.isfinite(probabilities).all() or
            not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Invalid classification records")
    if not set(truth) <= set(label_order) or not set(predictions) <= set(label_order):
        raise ValueError("Unexpected label")
    result = {"N": len(truth), "class_counts": dict(Counter(truth.tolist())),
              "accuracy": float(accuracy_score(truth, predictions)),
              "macro_F1": float(f1_score(truth, predictions, labels=list(label_order),
                                         average="macro", zero_division=0)),
              "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
              "AUROC": None, "AUROC_undefined_reason": None}
    if set(truth) != set(label_order) or set(classes) != set(label_order):
        result["AUROC_undefined_reason"] = "missing_true_or_trained_class"
    elif len(label_order) == 2:
        positive = list(classes).index("irreversible")
        result["AUROC"] = float(roc_auc_score(truth == "irreversible", probabilities[:, positive]))
    else:
        order = np.argsort(np.asarray(classes))
        result["AUROC"] = float(roc_auc_score(truth, probabilities[:, order],
            labels=np.asarray(classes)[order], multi_class="ovr", average="macro"))
    return result


def choose_linear(x_train, y_train, x_development, y_development, *, seed=1729,
                  lexical=False, label_order=LABEL_ORDER):
    """No test inputs are accepted. Tune macro-F1; resolve C ties downward."""
    y_train, y_development = np.asarray(y_train), np.asarray(y_development)
    if len(set(y_train)) < 2:
        raise ValueError("A logistic control with one training class is not estimable")
    if lexical:
        if len(x_train) != len(y_train) or len(x_development) != len(y_development):
            raise ValueError("Text/label lengths differ")
    else:
        x_train, x_development = np.asarray(x_train), np.asarray(x_development)
        if (x_train.ndim != 2 or x_development.ndim != 2 or
                x_train.shape[1] != x_development.shape[1] or
                len(x_train) != len(y_train) or len(x_development) != len(y_development) or
                not np.isfinite(x_train).all() or not np.isfinite(x_development).all()):
            raise ValueError("Invalid feature partitions")
    best, tried = None, []
    for strength in C_GRID:
        first = CountVectorizer(ngram_range=(1, 2)) if lexical else StandardScaler()
        pipeline = make_pipeline(first, LogisticRegression(
            C=strength, penalty="l2", solver="lbfgs", class_weight="balanced",
            max_iter=5000, random_state=seed))
        pipeline.fit(x_train, y_train)
        iterations = pipeline.named_steps["logisticregression"].n_iter_.tolist()
        if max(iterations) >= 5000:
            raise RuntimeError("Logistic fit reached the frozen iteration limit; retain the failed condition")
        predictions = pipeline.predict(x_development)
        measured = metrics(y_development, predictions, pipeline.predict_proba(x_development),
                           pipeline.classes_, label_order)
        tried.append({"C": strength, "development": measured, "iterations": iterations, "converged": True})
        if best is None or measured["macro_F1"] > best[0]:
            best = measured["macro_F1"], pipeline, strength, measured
    return best[1], {"selected_C": best[2], "development": best[3], "tried": tried,
                     "selection_metric": "development_macro_F1_then_smaller_C",
                     "test_data_used_for_selection": False}


def choose_layer(selections):
    if not selections:
        raise ValueError("No development layer selections")
    return min(selections, key=lambda layer: (-selections[layer]["development"]["balanced_accuracy"], int(layer)))


def training_majority(labels, label_order=LABEL_ORDER):
    counts = Counter(labels)
    if not counts:
        raise ValueError("Empty training labels")
    return min(label_order, key=lambda label: (-counts[label], label_order.index(label)))
