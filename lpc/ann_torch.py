"""PyTorch backprop for the same chain as ``lpc.ann.train_backprop``.

No biases. Hidden layers use the same σ. The descended objective is the batch
mean of ``1/2 ||y − ŷ||^2``, in float64, with plain SGD at step size ``eta``.
Reported errors are the entrywise mean of ``(y − ŷ)^2``.
"""

import numpy as np
import torch
from torch import nn

from lpc.activations import SUPPORTED


def train_backprop_torch(weights, x, y, steps, eta, activation):
    """Gradient descent for ``steps`` full-batch updates of size ``eta``.

    Copies ``weights`` into ``nn.Linear`` layers with ``bias=False``. Returns
    the entrywise MSE before and after every update (``steps + 1`` values)
    and the trained weights as NumPy arrays in the same ``(fan_out, fan_in)``
    layout.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if activation not in SUPPORTED:
        raise ValueError(
            "unsupported activation %r; expected one of %s" % (activation, ", ".join(SUPPORTED))
        )
    model = Chain(weights, activation)
    inputs = _tensor(x)
    targets = _tensor(y)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(eta))
    losses = [_entrywise_mse(model, inputs, targets)]
    for _ in range(steps):
        optimizer.zero_grad()
        prediction = model(inputs)
        loss = 0.5 * ((prediction - targets) ** 2).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        losses.append(_entrywise_mse(model, inputs, targets))
    trained = [layer.weight.detach().numpy().copy() for layer in model.layers]
    return losses, trained


class Chain(nn.Module):
    """Feedforward chain. The last map is linear. Hidden maps use σ."""

    def __init__(self, weights, activation):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        for weight in weights:
            fan_out, fan_in = np.asarray(weight).shape
            layer = nn.Linear(fan_in, fan_out, bias=False, dtype=torch.float64)
            with torch.no_grad():
                layer.weight.copy_(_tensor(weight))
            self.layers.append(layer)

    def forward(self, inputs):
        state = inputs
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            state = layer(state)
            if index < last:
                state = _activate(self.activation, state)
        return state


def _tensor(value):
    array = np.asarray(value, dtype=np.float64)
    return torch.tensor(array, dtype=torch.float64)


def _activate(name, pre):
    if name == "identity":
        return pre
    if name == "tanh":
        return torch.tanh(pre)
    if name == "relu":
        return torch.relu(pre)
    raise ValueError(
        "unsupported activation %r; expected one of %s" % (name, ", ".join(SUPPORTED))
    )


def _entrywise_mse(model, inputs, targets):
    with torch.no_grad():
        prediction = model(inputs)
        return float(torch.mean((prediction - targets) ** 2))
