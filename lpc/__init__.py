"""Augmented Lagrangian predictive coding (PC-ALM).

Faithful implementation of Algorithm 1 from Seely and Gould, arXiv:2605.31022.
"""

from lpc.inference import InferenceResult, infer
from lpc.network import Network
from lpc.train import train, train_step

__all__ = [
    "InferenceResult",
    "Network",
    "infer",
    "train",
    "train_step",
]
