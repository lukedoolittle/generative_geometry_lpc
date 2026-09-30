"""Learning rule and a tiny end-to-end training step."""

import numpy as np

from lpc.activations import sigma, sigma_prime
from lpc.energy import (
    activity_gradients,
    augmented_lagrangian,
    forward_hiddens,
    weight_gradients,
)
from lpc.network import Network
from lpc.train import train_step


def _backprop_gradients(weights, x, y, activation):
    """Batch-mean gradient of 1/2 ||y - W_L h_{L-1}||^2 through the Section 3 chain."""
    pres = []
    states = [x]
    for weight in weights[:-1]:
        pre = states[-1] @ weight.T
        pres.append(pre)
        states.append(sigma(activation, pre))
    prediction = states[-1] @ weights[-1].T
    batch = x.shape[0]
    output_error = prediction - y
    gradients = [None] * len(weights)
    gradients[-1] = (output_error.T @ states[-1]) / batch
    adjoint = output_error @ weights[-1]
    for index in range(len(weights) - 2, -1, -1):
        local = sigma_prime(activation, pres[index]) * adjoint
        gradients[index] = (local.T @ states[index]) / batch
        adjoint = local @ weights[index]
    return gradients


def test_activity_gradient_matches_energy_finite_difference():
    net = Network((2, 3, 4, 2), activation="tanh", seed=5, init_scale=0.4)
    rng = np.random.default_rng(6)
    x = rng.normal(size=(1, 2))
    y = rng.normal(size=(1, 2))
    hiddens = [rng.normal(scale=0.5, size=(1, width)) for width in net.sizes[1:-1]]
    multipliers = [rng.normal(scale=0.3, size=h.shape) for h in hiddens]
    analytic = activity_gradients(
        net.weights, hiddens, multipliers, x, y, rho=0.7, activation="tanh"
    )
    epsilon = 1e-6
    for layer, gradient in enumerate(analytic):
        numerical = np.zeros_like(gradient)
        for unit in range(gradient.shape[1]):
            stepped = [h.copy() for h in hiddens]
            stepped[layer][0, unit] += epsilon
            high = augmented_lagrangian(
                net.weights, stepped, multipliers, x, y, rho=0.7, activation="tanh"
            )
            stepped[layer][0, unit] -= 2.0 * epsilon
            low = augmented_lagrangian(
                net.weights, stepped, multipliers, x, y, rho=0.7, activation="tanh"
            )
            numerical[0, unit] = (high - low) / (2.0 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-6)


def test_weight_gradient_matches_energy_finite_difference():
    net = Network((2, 3, 2), activation="relu", seed=7, init_scale=0.5)
    rng = np.random.default_rng(8)
    x = rng.normal(size=(3, 2))
    y = rng.normal(size=(3, 2))
    hiddens = forward_hiddens(net.weights, x, "relu")
    # Shift off the kink so the squared penalty is differentiable here.
    hiddens = [h + 0.25 for h in hiddens]
    multipliers = [rng.normal(scale=0.2, size=h.shape) for h in hiddens]
    analytic = weight_gradients(
        net.weights, hiddens, multipliers, x, y, rho=1.2, activation="relu"
    )
    epsilon = 1e-6
    for index, gradient in enumerate(analytic):
        numerical = np.zeros_like(gradient)
        for row in range(gradient.shape[0]):
            for col in range(gradient.shape[1]):
                original = net.weights[index][row, col]
                net.weights[index][row, col] = original + epsilon
                high = augmented_lagrangian(
                    net.weights, hiddens, multipliers, x, y, rho=1.2, activation="relu"
                )
                net.weights[index][row, col] = original - epsilon
                low = augmented_lagrangian(
                    net.weights, hiddens, multipliers, x, y, rho=1.2, activation="relu"
                )
                net.weights[index][row, col] = original
                numerical[row, col] = (high - low) / (2.0 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-6)


def test_batch_weight_gradient_averages_per_sample_gradients():
    net = Network((2, 3, 1), activation="tanh", seed=9, init_scale=0.3)
    rng = np.random.default_rng(10)
    x = rng.normal(size=(4, 2))
    y = rng.normal(size=(4, 1))
    hiddens = [rng.normal(scale=0.4, size=(4, 3))]
    multipliers = [rng.normal(scale=0.2, size=(4, 3))]
    batched = weight_gradients(
        net.weights, hiddens, multipliers, x, y, rho=1.0, activation="tanh"
    )
    accumulated = [np.zeros_like(weight) for weight in net.weights]
    for sample in range(x.shape[0]):
        single = weight_gradients(
            net.weights,
            [hiddens[0][sample : sample + 1]],
            [multipliers[0][sample : sample + 1]],
            x[sample : sample + 1],
            y[sample : sample + 1],
            rho=1.0,
            activation="tanh",
        )
        for total, part in zip(accumulated, single):
            total += part
    for left, right in zip(batched, accumulated):
        np.testing.assert_allclose(left, right / x.shape[0])


def test_linear_equilibrium_weight_gradient_matches_backprop():
    net = Network((2, 4, 3, 2), activation="identity", seed=11, init_scale=0.3)
    rng = np.random.default_rng(12)
    x = rng.normal(size=(6, 2))
    y = rng.normal(size=(6, 2))
    result = net.infer(x, y, T=1500, eta_h=0.04, alpha=0.7, rho=1.0)
    learned = weight_gradients(
        net.weights,
        result.hiddens,
        result.multipliers,
        x,
        y,
        rho=1.0,
        activation="identity",
    )
    reference = _backprop_gradients(net.weights, x, y, "identity")
    for left, right in zip(learned, reference):
        np.testing.assert_allclose(left, right, atol=1e-6)


def test_tiny_training_step_runs_and_reduces_loss():
    teacher = Network((2, 4, 1), activation="identity", seed=13, init_scale=0.5)
    student = Network((2, 4, 1), activation="identity", seed=14, init_scale=0.15)
    rng = np.random.default_rng(15)
    x = rng.normal(size=(32, 2))
    y = teacher.predict(x)
    losses = student.train(
        x,
        y,
        steps=15,
        T=20,
        eta_h=0.08,
        alpha=1.0,
        rho=1.0,
        eta_theta=0.08,
    )
    assert len(losses) == 16
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0] * 0.5

    nonlinear = Network((2, 4, 1), activation="tanh", seed=16, init_scale=0.2)
    before = nonlinear.weights[0].copy()
    train_step(
        nonlinear.weights,
        x,
        np.tanh(y),
        activation="tanh",
        T=4,
        eta_h=0.05,
        alpha=0.5,
        rho=1.0,
        eta_theta=0.02,
    )
    assert not np.allclose(before, nonlinear.weights[0])
