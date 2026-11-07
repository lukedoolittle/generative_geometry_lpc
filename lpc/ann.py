"""Full-batch backprop for the same chain PC-ALM trains.

No biases, same elementwise σ. The descended objective is the batch mean of
``1/2 ||y − ŷ||^2``, the supervised term in the augmented Lagrangian.
Reported errors are the entrywise mean of ``(y − ŷ)^2``.
"""

import numpy as np

from lpc.activations import sigma, sigma_prime


def predict(weights, x, activation):
    """Forward map ŷ = W_L σ(W_{L-1} … σ(W_1 x))."""
    state = np.asarray(x, dtype=np.float64)
    for weight in weights[:-1]:
        state = sigma(activation, state @ weight.T)
    return state @ weights[-1].T


def forward_mse(weights, x, y, activation):
    """Entrywise mean squared error of ``predict``."""
    prediction = predict(weights, x, activation)
    return float(np.mean((prediction - np.asarray(y, dtype=np.float64)) ** 2))


def backprop_gradients(weights, x, y, activation):
    """Batch-mean gradient of mean ``1/2 ||y − ŷ||^2`` through the chain."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pres = []
    states = [x]
    for weight in weights[:-1]:
        pre = states[-1] @ weight.T
        pres.append(pre)
        states.append(sigma(activation, pre))
    prediction = states[-1] @ weights[-1].T
    batch = x.shape[0]
    error = prediction - y
    gradients = [None] * len(weights)
    gradients[-1] = (error.T @ states[-1]) / batch
    adjoint = error @ weights[-1]
    for index in range(len(weights) - 2, -1, -1):
        local = sigma_prime(activation, pres[index]) * adjoint
        gradients[index] = (local.T @ states[index]) / batch
        adjoint = local @ weights[index]
    return gradients


def train_backprop(weights, x, y, steps, eta, activation):
    """Gradient descent for ``steps`` full-batch updates of size ``eta``.

    Copies ``weights``. Returns the entrywise MSE before and after every
    update (``steps + 1`` values) and the trained weights.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    trained = [np.array(weight, dtype=np.float64, copy=True) for weight in weights]
    losses = [forward_mse(trained, x, y, activation)]
    for _ in range(steps):
        gradients = backprop_gradients(trained, x, y, activation)
        for weight, gradient in zip(trained, gradients):
            weight -= eta * gradient
        losses.append(forward_mse(trained, x, y, activation))
    return losses, trained
