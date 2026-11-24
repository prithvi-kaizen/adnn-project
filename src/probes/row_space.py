"""Fit the same standardized L2 objective in the training row span."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.probes.linear import C_GRID, metrics


def training_space(x_train, x_dev):
    scaler = StandardScaler().fit(x_train)
    train = scaler.transform(x_train).astype(np.float64)
    dev = scaler.transform(x_dev).astype(np.float64)
    eigenvalues, directions = np.linalg.eigh(train @ train.T)
    valid = eigenvalues > np.finfo(np.float64).eps * max(train.shape) * max(eigenvalues[-1], 1)
    if not valid.any():
        return scaler, np.zeros((train.shape[1], 1)), np.zeros((len(train), 1)), np.zeros((len(dev), 1))
    basis = (train.T @ directions[:, valid]) / np.sqrt(eigenvalues[valid])
    return scaler, basis, train @ basis, dev @ basis


def fit_in_span(space, y_train, y_dev, seed, order):
    scaler, basis, train, dev = space
    best, reports = None, []
    for strength in C_GRID:
        model = LogisticRegression(C=strength, penalty="l2", solver="lbfgs", class_weight="balanced",
                                   max_iter=5000, random_state=seed).fit(train, y_train)
        iterations = model.n_iter_.tolist()
        if max(iterations) >= 5000:
            raise RuntimeError("Reduced-coordinate optimizer reached the iteration limit")
        measured = metrics(y_dev, model.predict(dev), model.predict_proba(dev), model.classes_, order)
        reports.append({"C": strength, "development": measured, "iterations": iterations})
        if best is None or measured["macro_F1"] > best[0]:
            best = measured["macro_F1"], strength, measured, model
    _, strength, measured, model = best
    model.coef_ = model.coef_ @ basis.T
    model.n_features_in_ = basis.shape[0]
    return make_pipeline(scaler, model), {"selected_C": strength, "development": measured, "tried": reports,
                                         "training_span_dimension": int(basis.shape[1])}
