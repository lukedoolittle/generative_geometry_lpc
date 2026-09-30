"""Augmented Lagrangian of a feedforward chain and its local gradients.

The energy is equation (5) of Seely and Gould (arXiv:2605.31022). Activity
gradients are equation (11) for hidden layers 1..L-2, plus the supervised
boundary term for layer L-1 from Appendix A. Weight gradients are
∇_θ L_ρ, the learning rule in equation (13).
"""

import numpy as np

from lpc.activations import sigma, sigma_prime


def as_batch(x, y):
    """Promote a single sample to a batch of one. Returns float64 arrays."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must have shape (batch, features)")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must share the batch dimension")
    return x, y


def validate_weights(weights, x, y):
    if len(weights) < 2:
        raise ValueError("need depth L >= 2 (at least one hidden layer and a readout)")
    batch = x.shape[0]
    if weights[0].shape[1] != x.shape[1]:
        raise ValueError(
            "input width %s does not match W_1 shape %s"
            % (x.shape[1], weights[0].shape)
        )
    if weights[-1].shape[0] != y.shape[1]:
        raise ValueError(
            "output width %s does not match W_L shape %s"
            % (y.shape[1], weights[-1].shape)
        )
    for i in range(len(weights) - 1):
        if weights[i].shape[0] != weights[i + 1].shape[1]:
            raise ValueError(
                "W_%d shape %s does not feed W_%d shape %s"
                % (i + 1, weights[i].shape, i + 2, weights[i + 1].shape)
            )
        if weights[i].ndim != 2 or weights[i + 1].ndim != 2:
            raise ValueError("each weight must be a matrix")
    for W in weights:
        if W.shape[0] < 1 or W.shape[1] < 1:
            raise ValueError("weight matrices must be non-empty")
    return batch


def forward_hiddens(weights, x, activation):
    """Forward pass h_i = σ(W_i h_{i-1}) for i = 1..L-1. h_0 = x is clamped.

    Returns the free hidden states only. This is the initialization in
    Algorithm 1 and the feasible point of the constraints in equation (1).
    The readout is linear and is not included.
    """
    hiddens = []
    previous = x
    for weight in weights[:-1]:
        previous = sigma(activation, previous @ weight.T)
        hiddens.append(previous)
    return hiddens


def predict(weights, x, activation):
    """Linear readout ŷ = W_L h_{L-1} after a feasible forward pass."""
    x = np.asarray(x, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)
    hiddens = forward_hiddens(weights, x, activation)
    prediction = hiddens[-1] @ weights[-1].T
    if single:
        return prediction[0]
    return prediction


def supervised_mse(weights, x, y, activation):
    """Mean squared error of the feasible forward readout, mean over entries."""
    x, y = as_batch(x, y)
    prediction = predict(weights, x, activation)
    return float(np.mean((prediction - y) ** 2))


def layer_residuals(weights, hiddens, x, activation):
    """r_i = h_i - σ(W_i h_{i-1}) for i = 1..L-1."""
    states = [x, *hiddens]
    residuals = []
    for index, hidden in enumerate(hiddens):
        pre = states[index] @ weights[index].T
        residuals.append(hidden - sigma(activation, pre))
    return residuals


def _preactivations(weights, hiddens, x):
    states = [x, *hiddens]
    return [states[index] @ weights[index].T for index in range(len(hiddens))]


def augmented_lagrangian(weights, hiddens, multipliers, x, y, rho, activation):
    """Mean over the batch of L_ρ (equation 5).

    Per sample:

        1/2 ||y - W_L h_{L-1}||^2
        + Σ_i λ_i^T (h_i - σ(W_i h_{i-1}))
        + (ρ/2) Σ_i ||h_i - σ(W_i h_{i-1})||^2
    """
    x, y = as_batch(x, y)
    validate_weights(weights, x, y)
    prediction = hiddens[-1] @ weights[-1].T
    per_sample = 0.5 * np.sum((y - prediction) ** 2, axis=1)
    residuals = layer_residuals(weights, hiddens, x, activation)
    for residual, multiplier in zip(residuals, multipliers):
        per_sample = per_sample + np.sum(multiplier * residual, axis=1)
        per_sample = per_sample + 0.5 * rho * np.sum(residual * residual, axis=1)
    return float(np.mean(per_sample))


def activity_gradients(weights, hiddens, multipliers, x, y, rho, activation):
    """Per-sample ∇_{h_i} L_ρ. Not averaged over the batch.

    For 1 ≤ i ≤ L-2 this is equation (11):

        (ρ r_i + λ_i) - W_{i+1}^T diag(σ'(W_{i+1} h_i)) (ρ r_{i+1} + λ_{i+1})

    For the top hidden state i = L-1 the next factor is the linear readout,
    so Appendix A supplies the supervised term instead of another constraint:

        ∇_{h_{L-1}} L_ρ = -W_L^T (y - W_L h_{L-1}) + (ρ r_{L-1} + λ_{L-1})
    """
    x, y = as_batch(x, y)
    validate_weights(weights, x, y)
    depth = len(weights)
    residuals = layer_residuals(weights, hiddens, x, activation)
    pres = _preactivations(weights, hiddens, x)
    prediction = hiddens[-1] @ weights[-1].T
    supervised_residual = y - prediction
    gradients = []
    for layer in range(1, depth):
        composite = multipliers[layer - 1] + rho * residuals[layer - 1]
        if layer < depth - 1:
            slope = sigma_prime(activation, pres[layer])
            next_composite = multipliers[layer] + rho * residuals[layer]
            pulled = (slope * next_composite) @ weights[layer]
            gradients.append(composite - pulled)
        else:
            gradients.append(composite - supervised_residual @ weights[-1])
    return gradients


def weight_gradients(weights, hiddens, multipliers, x, y, rho, activation):
    """Batch-mean ∇_{W_i} L_ρ, the summand of the Algorithm 1 learning step.

    For hidden layers i = 1..L-1, differentiating equation (5) gives

        ∇_{W_i} L_ρ = -diag(σ'(W_i h_{i-1})) (λ_i + ρ r_i) h_{i-1}^T

    The linear readout does not appear in a constraint:

        ∇_{W_L} L_ρ = -(y - W_L h_{L-1}) h_{L-1}^T

    Each returned matrix is the mean of those per-sample gradients.
    """
    x, y = as_batch(x, y)
    validate_weights(weights, x, y)
    batch = x.shape[0]
    states = [x, *hiddens]
    gradients = []
    for layer in range(1, len(weights)):
        pre = states[layer - 1] @ weights[layer - 1].T
        residual = states[layer] - sigma(activation, pre)
        slope = sigma_prime(activation, pre)
        signal = slope * (multipliers[layer - 1] + rho * residual)
        gradients.append(-(signal.T @ states[layer - 1]) / batch)
    prediction = hiddens[-1] @ weights[-1].T
    readout_error = prediction - y
    gradients.append((readout_error.T @ hiddens[-1]) / batch)
    return gradients
