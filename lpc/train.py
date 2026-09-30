"""PC-ALM learning rule, equation (13) and Algorithm 1 line 12."""

from lpc.energy import as_batch, supervised_mse, weight_gradients
from lpc.inference import infer


def train_step(weights, x, y, activation, T, eta_h, alpha, rho, eta_theta):
    """One batch: infer, then descend on ∇_W L_ρ averaged over the batch.

    Weights are updated in place. Returns the inference state that produced
    the weight gradient.
    """
    x, y = as_batch(x, y)
    result = infer(
        weights,
        x,
        y,
        activation=activation,
        T=T,
        eta_h=eta_h,
        alpha=alpha,
        rho=rho,
    )
    gradients = weight_gradients(
        weights,
        result.hiddens,
        result.multipliers,
        x,
        y,
        rho,
        activation,
    )
    for weight, gradient in zip(weights, gradients):
        weight -= eta_theta * gradient
    return result


def train(weights, x, y, activation, steps, T, eta_h, alpha, rho, eta_theta):
    """Repeat ``train_step`` and record feasible-forward MSE before and after.

    The recorded loss is the supervised objective of equation (1) evaluated
    on the forward pass, which is the quantity the KKT weight gradient
    descends. The list has ``steps + 1`` entries.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    x, y = as_batch(x, y)
    losses = [supervised_mse(weights, x, y, activation)]
    for _ in range(steps):
        train_step(
            weights,
            x,
            y,
            activation=activation,
            T=T,
            eta_h=eta_h,
            alpha=alpha,
            rho=rho,
            eta_theta=eta_theta,
        )
        losses.append(supervised_mse(weights, x, y, activation))
    return losses
