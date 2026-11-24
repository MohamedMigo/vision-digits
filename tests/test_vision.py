import tempfile

import numpy as np
from sklearn.datasets import load_digits

from vision.train import augment, gradient_features, main


def test_features_shape():
    X = load_digits().data[:5]
    assert gradient_features(X).shape == (5, 64 + 64 + 16)


def test_augment_triples():
    d = load_digits()
    X, y = augment(d.data[:10], d.target[:10])
    assert len(X) == 30 and len(y) == 30 and np.array_equal(X[:10], d.data[:10])


def test_accuracy():
    with tempfile.TemporaryDirectory() as t:
        assert max(main(t)["accuracy"].values()) > 0.95
