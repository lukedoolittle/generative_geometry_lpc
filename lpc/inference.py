"""PC-ALM inference: finite-budget primal-dual dynamics (Algorithm 1)."""

from dataclasses import dataclass

import numpy as np

from lpc.energy import (
    as_batch,
    augmented_lagrangian,
    forward_sweep,
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
    """Run Algorithm 1 for one batch, with the dual step inside the forward sweep.

    Every iteration is one input-to-output sweep. The first sweep writes the
    feasible activities h_i = σ(W_i h_{i-1}) and applies λ ← λ + α r to a zero
    residual, so each multiplier starts at 0. The next T - 1 sweeps take a
    primal step and, in that same layer loop, the dual step

        λ ← λ + α r,    r = h - σ(W h_prev)

    using the residual of the activities just written and the combined signal
    c = λ + ρ r inside ∇_h L_ρ. The last sweep is primal only. Multipliers are
    inference state: they are not kept across calls.
    """
    if T < 1:
        raise ValueError("inference budget T must be at least 1")
    x, y = as_batch(x, y)
    validate_weights(weights, x, y)
    hiddens, multipliers = forward_sweep(
        weights,
        x,
        activation,
        y=y,
        alpha=alpha,
        rho=rho,
        update_multipliers=True,
    )
    for _ in range(T - 1):
        hiddens, multipliers = forward_sweep(
            weights,
            x,
            activation,
            hiddens=hiddens,
            multipliers=multipliers,
            y=y,
            eta_h=eta_h,
            alpha=alpha,
            rho=rho,
            update_activities=True,
            update_multipliers=True,
        )
    hiddens, multipliers = forward_sweep(
        weights,
        x,
        activation,
        hiddens=hiddens,
        multipliers=multipliers,
        y=y,
        eta_h=eta_h,
        alpha=alpha,
        rho=rho,
        update_activities=True,
        update_multipliers=False,
    )
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
