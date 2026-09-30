"""Small feedforward chain whose weights PC-ALM trains."""

import numpy as np


class Network:
    """Depth-L chain from Section 3.

    ``sizes`` is (n_0, n_1, ..., n_L). Hidden layers use h_i = σ(W_i h_{i-1})
    and the readout is the linear map W_L h_{L-1}. There are no biases, matching
    the paper. Each W_i is stored with shape (n_i, n_{i-1}).
    """

    def __init__(self, sizes, activation="identity", seed=0, init_scale=0.1):
        if len(sizes) < 3:
            raise ValueError("sizes must include input, one hidden layer, and output")
        self.sizes = tuple(int(size) for size in sizes)
        self.activation = activation
        self.init_scale = float(init_scale)
        rng = np.random.default_rng(seed)
        self.weights = []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            self.weights.append(
                rng.normal(0.0, self.init_scale, size=(fan_out, fan_in))
            )

    def predict(self, x):
        from lpc.energy import predict

        return predict(self.weights, x, self.activation)

    def infer(self, x, y, T, eta_h, alpha, rho):
        from lpc.inference import infer

        return infer(
            self.weights,
            x,
            y,
            activation=self.activation,
            T=T,
            eta_h=eta_h,
            alpha=alpha,
            rho=rho,
        )

    def train(self, x, y, steps, T, eta_h, alpha, rho, eta_theta):
        from lpc.train import train

        return train(
            self.weights,
            x,
            y,
            activation=self.activation,
            steps=steps,
            T=T,
            eta_h=eta_h,
            alpha=alpha,
            rho=rho,
            eta_theta=eta_theta,
        )
