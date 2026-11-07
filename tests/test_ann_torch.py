"""PyTorch backprop matches the NumPy ANN on the same weights and batch."""

import numpy as np

from lpc.ann import train_backprop
from lpc.ann_torch import train_backprop_torch


def test_torch_full_batch_matches_numpy_on_each_activation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 5))
    y = rng.normal(size=(12, 2))
    weights = [
        rng.normal(scale=0.2, size=(7, 5)),
        rng.normal(scale=0.2, size=(4, 7)),
        rng.normal(scale=0.2, size=(2, 4)),
    ]
    for activation in ("identity", "tanh", "relu"):
        numpy_losses, numpy_weights = train_backprop(
            weights, x, y, steps=5, eta=0.08, activation=activation
        )
        torch_losses, torch_weights = train_backprop_torch(
            weights, x, y, steps=5, eta=0.08, activation=activation
        )
        np.testing.assert_allclose(torch_losses, numpy_losses, rtol=1e-10, atol=1e-10)
        for left, right in zip(torch_weights, numpy_weights):
            np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10)
