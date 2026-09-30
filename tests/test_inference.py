"""Inference dynamics of PC-ALM, Algorithm 1."""

import numpy as np

from lpc.energy import activity_gradients, augmented_lagrangian, forward_hiddens
from lpc.inference import infer
from lpc.network import Network


def _scalar_weights():
    """Figure 1 scalar chain: y_hat = w2 * w1 * x, with one free hidden unit."""
    weights = [np.array([[0.5]]), np.array([[1.5]])]
    x = np.array([[1.0]])
    y = np.array([[1.0]])
    return weights, x, y


def test_primal_step_with_zero_multiplier_is_predictive_coding():
    """α = 0 pins λ at 0, and T = 1 never applies a dual update."""
    net = Network((3, 4, 2), activation="tanh", seed=0, init_scale=0.2)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 3))
    y = rng.normal(size=(5, 2))
    short = net.infer(x, y, T=1, eta_h=0.05, alpha=1.0, rho=1.0)
    pinned = net.infer(x, y, T=6, eta_h=0.05, alpha=0.0, rho=1.0)
    assert all(np.all(layer == 0.0) for layer in short.multipliers)
    assert all(np.all(layer == 0.0) for layer in pinned.multipliers)


def test_dual_step_accumulates_the_residual():
    """Equation (12): after the first primal step, λ = α r(h_updated)."""
    weights, x, y = _scalar_weights()
    alpha = 0.4
    result = infer(
        weights, x, y, activation="identity", T=2, eta_h=0.05, alpha=alpha, rho=1.0
    )
    # T = 2 is one primal-dual cycle plus a final primal step, so λ was
    # written from the residual *before* the final primal step. Re-run the
    # first cycle explicitly and compare λ.
    hiddens = forward_hiddens(weights, x, "identity")
    multipliers = [np.zeros_like(h) for h in hiddens]
    gradients = activity_gradients(
        weights, hiddens, multipliers, x, y, rho=1.0, activation="identity"
    )
    hiddens = [h - 0.05 * g for h, g in zip(hiddens, gradients)]
    residual = hiddens[0] - (x @ weights[0].T)
    expected = alpha * residual
    np.testing.assert_allclose(result.multipliers[0], expected)


def test_one_primal_step_decreases_the_augmented_lagrangian():
    net = Network((2, 5, 3, 2), activation="tanh", seed=3, init_scale=0.3)
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4, 2))
    y = rng.normal(size=(4, 2))
    hiddens = forward_hiddens(net.weights, x, net.activation)
    multipliers = [rng.normal(scale=0.2, size=h.shape) for h in hiddens]
    before = augmented_lagrangian(
        net.weights, hiddens, multipliers, x, y, rho=1.0, activation="tanh"
    )
    gradients = activity_gradients(
        net.weights, hiddens, multipliers, x, y, rho=1.0, activation="tanh"
    )
    eta = 1e-3
    stepped = [h - eta * g for h, g in zip(hiddens, gradients)]
    after = augmented_lagrangian(
        net.weights, stepped, multipliers, x, y, rho=1.0, activation="tanh"
    )
    assert after < before


def test_linear_scalar_inference_reaches_the_backprop_adjoint():
    """Figure 1 / Proposition 3: h returns to the forward pass and λ → BP adjoint.

    With α = 0 the same activity step is penalty PC, which settles at a
    different hidden state. That separates PC-ALM from generic PC.
    """
    weights, x, y = _scalar_weights()
    forward_h = float((x @ weights[0].T)[0, 0])
    w2 = float(weights[1][0, 0])
    # λ_∞ = (y - w2 h) w2, Appendix A with the paper's residual sign.
    adjoint = (float(y[0, 0]) - w2 * forward_h) * w2

    alm = infer(
        weights,
        x,
        y,
        activation="identity",
        T=400,
        eta_h=0.05,
        alpha=0.8,
        rho=1.0,
    )
    np.testing.assert_allclose(alm.hiddens[0], [[forward_h]], atol=1e-5)
    np.testing.assert_allclose(alm.multipliers[0], [[adjoint]], atol=1e-4)
    np.testing.assert_allclose(alm.residuals[0], [[0.0]], atol=1e-5)

    pc = infer(
        weights,
        x,
        y,
        activation="identity",
        T=400,
        eta_h=0.05,
        alpha=0.0,
        rho=1.0,
    )
    # Stationarity of equation (2) on this scalar: h = 8/13, not w1 x.
    np.testing.assert_allclose(pc.hiddens[0], [[8.0 / 13.0]], atol=1e-4)
    assert abs(float(pc.hiddens[0][0, 0]) - forward_h) > 0.05


def test_longer_inference_drives_the_linear_residual_down():
    weights, x, y = _scalar_weights()
    short = infer(
        weights, x, y, activation="identity", T=1, eta_h=0.05, alpha=0.8, rho=1.0
    )
    long = infer(
        weights, x, y, activation="identity", T=200, eta_h=0.05, alpha=0.8, rho=1.0
    )
    short_residual = abs(float(np.asarray(short.residuals[0]).reshape(-1)[0]))
    long_residual = abs(float(np.asarray(long.residuals[0]).reshape(-1)[0]))
    assert short_residual > 1e-3
    assert long_residual < short_residual * 1e-3
