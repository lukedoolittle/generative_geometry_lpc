"""PC-ALM inference: finite-budget primal-dual dynamics (Algorithm 1)."""

from dataclasses import dataclass

import numpy as np

from lpc.energy import (
    activity_gradients,
    as_batch,
    augmented_lagrangian,
    forward_hiddens,
    layer_residuals,
    validate_weights,
)


@dataclass
class InferenceResult:
    """State after the inference loop, before the weight update."""

    hiddens: list
    multipliers: list
    residuals: list
    energy: float
    prediction: np.ndarray


def infer(weights, x, y, activation, T, eta_h, alpha, rho):
    """Run Algorithm 1 lines 4–10 for one batch.

    Hidden states are initialized on the forward pass and every multiplier
    starts at 0. The loop then takes T-1 primal-dual cycles

        h ← h - η_h ∇_h L_ρ
        λ ← λ + α (h - σ(W h_prev))

    and one final primal step with no trailing dual update. The dual step
    uses the residual of the updated activities, matching equation (16).
    Multipliers are inference state: they are not kept across calls.
    """
    if T < 1:
        raise ValueError("inference budget T must be at least 1")
    x, y = as_batch(x, y)
    validate_weights(weights, x, y)
    hiddens = forward_hiddens(weights, x, activation)
    multipliers = [np.zeros_like(hidden) for hidden in hiddens]
    for _ in range(T - 1):
        gradients = activity_gradients(
            weights, hiddens, multipliers, x, y, rho, activation
        )
        hiddens = [
            hidden - eta_h * gradient
            for hidden, gradient in zip(hiddens, gradients)
        ]
        residuals = layer_residuals(weights, hiddens, x, activation)
        multipliers = [
            multiplier + alpha * residual
            for multiplier, residual in zip(multipliers, residuals)
        ]
    gradients = activity_gradients(
        weights, hiddens, multipliers, x, y, rho, activation
    )
    hiddens = [
        hidden - eta_h * gradient for hidden, gradient in zip(hiddens, gradients)
    ]
    residuals = layer_residuals(weights, hiddens, x, activation)
    energy = augmented_lagrangian(
        weights, hiddens, multipliers, x, y, rho, activation
    )
    prediction = hiddens[-1] @ weights[-1].T
    return InferenceResult(
        hiddens=hiddens,
        multipliers=multipliers,
        residuals=residuals,
        energy=energy,
        prediction=prediction,
    )
