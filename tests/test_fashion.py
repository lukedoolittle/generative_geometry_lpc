"""Fashion-MNIST helpers and a tiny one-hot training check. No download."""

import gzip
import struct

import numpy as np

from lpc.ann import train_backprop
from lpc.fashion import fashion_split
from lpc.fashion_mnist import (
    accuracy,
    average_pool,
    one_hot,
    read_idx_images,
    read_idx_labels,
    stratified_indices,
)
from lpc.network import Network


def _gzip_bytes(payload):
    return gzip.compress(payload)


def test_idx_images_and_labels_roundtrip(tmp_path):
    images = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    labels = np.array([1, 9], dtype=np.uint8)
    image_path = tmp_path / "images.gz"
    label_path = tmp_path / "labels.gz"
    image_header = struct.pack(">IIII", 2051, 2, 3, 4)
    label_header = struct.pack(">II", 2049, 2)
    image_path.write_bytes(_gzip_bytes(image_header + images.tobytes()))
    label_path.write_bytes(_gzip_bytes(label_header + labels.tobytes()))
    np.testing.assert_array_equal(read_idx_images(str(image_path)), images)
    np.testing.assert_array_equal(read_idx_labels(str(label_path)), labels)


def test_stratified_sample_is_balanced_and_one_hot_matches_labels():
    labels = np.repeat(np.arange(4), 6)
    indices = stratified_indices(labels, per_class=3, rng=np.random.default_rng(0))
    chosen = labels[indices]
    assert indices.shape == (12,)
    assert len(set(indices.tolist())) == 12
    for class_index in range(4):
        assert int(np.sum(chosen == class_index)) == 3
    targets = one_hot(chosen, 4)
    assert targets.shape == (12, 4)
    np.testing.assert_array_equal(targets.argmax(axis=1), chosen)


def test_average_pool_means_each_block():
    images = np.arange(16, dtype=np.float64).reshape(1, 4, 4)
    pooled = average_pool(images, 2)
    assert pooled.shape == (1, 2, 2)
    np.testing.assert_allclose(pooled[0, 0, 0], np.mean([0, 1, 4, 5]))
    np.testing.assert_allclose(pooled[0, 1, 1], np.mean([10, 11, 14, 15]))


def test_split_flattens_a_fake_gallery():
    dataset = {
        "train_images": np.zeros((20, 2, 2), dtype=np.float64),
        "train_labels": np.repeat(np.arange(2), 10),
        "test_images": np.ones((8, 2, 2), dtype=np.float64),
        "test_labels": np.repeat(np.arange(2), 4),
    }
    split = fashion_split(dataset, train_per_class=3, test_per_class=2, train_seed=0, test_seed=1)
    assert split["x_train"].shape == (6, 4)
    assert split["x_test"].shape == (4, 4)
    assert set(split["y_train"].tolist()) == {0, 1}


def test_one_hot_training_beats_chance_on_separated_classes():
    rng = np.random.default_rng(0)
    means = np.array([[-1.5, 0.0], [1.5, 0.0], [0.0, 1.5]])
    samples = []
    labels = []
    for class_index, mean in enumerate(means):
        samples.append(mean + rng.normal(scale=0.15, size=(24, 2)))
        labels.append(np.full(24, class_index))
    x = np.vstack(samples)
    y = np.concatenate(labels)
    targets = one_hot(y, 3)
    net = Network((2, 8, 3), activation="tanh", seed=1, init_scale=0.4)
    ann_init = [weight.copy() for weight in net.weights]
    before = accuracy(net.predict(x), y)
    losses = net.train(x, targets, steps=30, T=6, eta_h=0.1, alpha=1.0, rho=1.0, eta_theta=0.2)
    ann_losses, ann_weights = train_backprop(
        ann_init, x, targets, steps=30, eta=0.2, activation="tanh"
    )
    after = accuracy(net.predict(x), y)
    assert losses[-1] < losses[0]
    assert ann_losses[-1] < ann_losses[0]
    assert after > before
    assert after > 0.8
    assert ann_weights[0].shape == (8, 2)
