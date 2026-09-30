"""Train and run inference on a small synthetic regression problem.

Usage, from the repository root:

    python -m lpc
"""

import numpy as np

from lpc.network import Network


def _mse(prediction, target):
    return float(np.mean((prediction - target) ** 2))


def main():
    rng = np.random.default_rng(0)
    sizes = (2, 8, 1)
    teacher = Network(sizes, activation="identity", seed=1, init_scale=0.6)
    student = Network(sizes, activation="identity", seed=2, init_scale=0.15)
    inputs = rng.normal(size=(64, 2))
    targets = teacher.predict(inputs)

    hyper = dict(T=20, eta_h=0.1, alpha=1.0, rho=1.0, eta_theta=0.15)
    losses = student.train(inputs, targets, steps=40, **hyper)
    heldout_x = rng.normal(size=(16, 2))
    heldout_y = teacher.predict(heldout_x)
    inferred = student.infer(heldout_x, heldout_y, T=hyper["T"], eta_h=hyper["eta_h"], alpha=hyper["alpha"], rho=hyper["rho"])
    residual = float(
        np.mean([np.mean(np.abs(layer)) for layer in inferred.residuals])
    )

    print("PC-ALM synthetic regression (identity chain, sizes %s)" % (sizes,))
    print("forward MSE before training: %.6f" % losses[0])
    print("forward MSE after training:  %.6f" % losses[-1])
    print("held-out forward MSE:         %.6f" % _mse(student.predict(heldout_x), heldout_y))
    print("inference energy L_rho:       %.6f" % inferred.energy)
    print("mean |residual| after inference: %.6e" % residual)
    print("prediction[0]: %.4f  target[0]: %.4f" % (inferred.prediction[0, 0], heldout_y[0, 0]))

    if not np.isfinite(losses[-1]) or not np.isfinite(inferred.energy):
        return 1
    if not losses[-1] < losses[0]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
