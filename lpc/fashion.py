"""Fashion-MNIST classification with the existing squared-error chain.

Usage, from the repository root:

    python -m lpc.fashion

Targets are one-hot class vectors. The loss is the same feasible-forward
mean squared error as the identity-chain demo. Accuracy is the argmax of
the linear readout. ``python -m lpc`` is unchanged.

The split is stratified: 200 training images and 100 test images from each
of the 10 classes, seeds 0 and 1. The network is ``(784, 64, 10)`` with
``tanh`` and Gaussian weights of standard deviation ``1/sqrt(784)``.
"""

import numpy as np

from lpc.ann import predict, train_backprop
from lpc.ann_torch import train_backprop_torch
from lpc.fashion_mnist import accuracy, load_fashion_mnist, one_hot, stratified_indices
from lpc.network import Network

TRAIN_PER_CLASS = 200
TEST_PER_CLASS = 100
SIZES = (784, 64, 10)
ACTIVATION = "tanh"
STEPS = 40
T = 8
ETA_H = 0.1
ALPHA = 1.0
RHO = 1.0
ETA_THETA = 0.15
N_CLASSES = 10


def fashion_split(dataset, train_per_class, test_per_class, train_seed, test_seed):
    """Flatten a stratified train and test slice. Pixels are already in ``[0, 1]``."""
    train_index = stratified_indices(
        dataset["train_labels"], train_per_class, np.random.default_rng(train_seed)
    )
    test_index = stratified_indices(
        dataset["test_labels"], test_per_class, np.random.default_rng(test_seed)
    )
    train_images = dataset["train_images"][train_index]
    test_images = dataset["test_images"][test_index]
    return {
        "x_train": train_images.reshape(train_images.shape[0], -1),
        "y_train": dataset["train_labels"][train_index],
        "x_test": test_images.reshape(test_images.shape[0], -1),
        "y_test": dataset["test_labels"][test_index],
    }


def main():
    dataset = load_fashion_mnist()
    split = fashion_split(dataset, TRAIN_PER_CLASS, TEST_PER_CLASS, train_seed=0, test_seed=1)
    targets = one_hot(split["y_train"], N_CLASSES)
    test_targets = one_hot(split["y_test"], N_CLASSES)
    scale = 1.0 / np.sqrt(SIZES[0])
    student = Network(SIZES, activation=ACTIVATION, seed=2, init_scale=scale)
    ann_init = [weight.copy() for weight in student.weights]

    losses = student.train(
        split["x_train"],
        targets,
        steps=STEPS,
        T=T,
        eta_h=ETA_H,
        alpha=ALPHA,
        rho=RHO,
        eta_theta=ETA_THETA,
    )
    ann_losses, ann_weights = train_backprop(
        ann_init,
        split["x_train"],
        targets,
        steps=STEPS,
        eta=ETA_THETA,
        activation=ACTIVATION,
    )
    torch_losses, torch_weights = train_backprop_torch(
        ann_init,
        split["x_train"],
        targets,
        steps=STEPS,
        eta=ETA_THETA,
        activation=ACTIVATION,
    )
    pc_train = student.predict(split["x_train"])
    pc_test = student.predict(split["x_test"])
    ann_train = predict(ann_weights, split["x_train"], ACTIVATION)
    ann_test = predict(ann_weights, split["x_test"], ACTIVATION)
    torch_train = predict(torch_weights, split["x_train"], ACTIVATION)
    torch_test = predict(torch_weights, split["x_test"], ACTIVATION)
    pc_train_acc = accuracy(pc_train, split["y_train"])
    pc_test_acc = accuracy(pc_test, split["y_test"])
    ann_train_acc = accuracy(ann_train, split["y_train"])
    ann_test_acc = accuracy(ann_test, split["y_test"])
    torch_train_acc = accuracy(torch_train, split["y_train"])
    torch_test_acc = accuracy(torch_test, split["y_test"])
    ann_test_mse = float(np.mean((ann_test - test_targets) ** 2))
    pc_test_mse = float(np.mean((pc_test - test_targets) ** 2))
    torch_test_mse = float(np.mean((torch_test - test_targets) ** 2))

    print("Fashion-MNIST classification (one-hot squared error, sizes %s)" % (SIZES,))
    print("pixels: 28x28 scaled to [0, 1], flattened")
    print(
        "samples: %d train (%d per class), %d test (%d per class)"
        % (split["x_train"].shape[0], TRAIN_PER_CLASS, split["x_test"].shape[0], TEST_PER_CLASS)
    )
    print("activation: %s    inference budget T: %d    steps: %d" % (ACTIVATION, T, STEPS))
    print("PC-ALM train MSE:      %.6f -> %.6f" % (losses[0], losses[-1]))
    print("ANN NumPy train MSE:   %.6f -> %.6f" % (ann_losses[0], ann_losses[-1]))
    print("ANN PyTorch train MSE: %.6f -> %.6f" % (torch_losses[0], torch_losses[-1]))
    print("PC-ALM test MSE:       %.6f" % pc_test_mse)
    print("ANN NumPy test MSE:    %.6f" % ann_test_mse)
    print("ANN PyTorch test MSE:  %.6f" % torch_test_mse)
    print("PC-ALM train accuracy:      %.3f" % pc_train_acc)
    print("ANN NumPy train accuracy:   %.3f" % ann_train_acc)
    print("ANN PyTorch train accuracy: %.3f" % torch_train_acc)
    print("PC-ALM test accuracy:       %.3f" % pc_test_acc)
    print("ANN NumPy test accuracy:    %.3f" % ann_test_acc)
    print("ANN PyTorch test accuracy:  %.3f" % torch_test_acc)

    reported = (
        losses[-1],
        ann_losses[-1],
        torch_losses[-1],
        pc_test_mse,
        ann_test_mse,
        torch_test_mse,
    )
    if not all(np.isfinite(value) for value in reported):
        return 1
    if not (losses[-1] < losses[0] and ann_losses[-1] < ann_losses[0] and torch_losses[-1] < torch_losses[0]):
        return 1
    if min(pc_train_acc, pc_test_acc, ann_train_acc, ann_test_acc, torch_train_acc, torch_test_acc) <= 0.5:
        return 1
    if not np.allclose(torch_losses, ann_losses, rtol=1e-8, atol=1e-8):
        return 1
    if torch_train_acc != ann_train_acc or torch_test_acc != ann_test_acc:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
