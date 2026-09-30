"""Elementwise nonlinearities σ from the Section 3 chain."""

import numpy as np

SUPPORTED = ("identity", "tanh", "relu")


def sigma(name, pre):
    """Apply σ. ``pre`` is the pre-activation W h."""
    if name == "identity":
        return pre
    if name == "tanh":
        return np.tanh(pre)
    if name == "relu":
        return np.maximum(pre, 0.0)
    raise ValueError(
        "unsupported activation %r; expected one of %s" % (name, ", ".join(SUPPORTED))
    )


def sigma_prime(name, pre):
    """Elementwise σ′(pre), the diagonal of the activation Jacobian."""
    if name == "identity":
        return np.ones_like(pre)
    if name == "tanh":
        value = np.tanh(pre)
        return 1.0 - value * value
    if name == "relu":
        return (pre > 0.0).astype(pre.dtype, copy=False)
    raise ValueError(
        "unsupported activation %r; expected one of %s" % (name, ", ".join(SUPPORTED))
    )
