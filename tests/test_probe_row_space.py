import numpy as np
import pytest

from src.probes.linear import LABEL_ORDER, choose_linear
from src.probes.row_space import fit_in_span, training_space


@pytest.mark.parametrize("order", [("reversible", "irreversible"), LABEL_ORDER])
def test_row_space_retains_linear_probe_predictions(order):
    rng = np.random.default_rng(1729)
    y = np.resize(np.array(order), 48)
    source = rng.normal(size=(48, 9)) + np.array([order.index(item) for item in y])[:, None]
    matrix = (source @ rng.normal(size=(9, 64))).astype(np.float32)
    train, dev = matrix[:36], matrix[36:]
    expected, baseline = choose_linear(train, y[:36], dev, y[36:], label_order=order)
    actual, fit = fit_in_span(training_space(train, dev), y[:36], y[36:], 1729, order)
    assert fit["selected_C"] == baseline["selected_C"]
    np.testing.assert_array_equal(actual.predict(dev), expected.predict(dev))
    np.testing.assert_allclose(actual.predict_proba(dev), expected.predict_proba(dev), atol=2e-3)
