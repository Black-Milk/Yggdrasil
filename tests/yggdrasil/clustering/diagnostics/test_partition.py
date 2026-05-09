import numpy as np
import pytest
from sklearn.datasets import make_blobs

from yggdrasil.clustering.diagnostics.partition import (
    modularity_on_kernel,
    silhouette_on_embedding,
)


def test_silhouette_high_for_well_separated_blobs():
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(0.0, 0.05, (10, 2)), rng.normal(5.0, 0.05, (10, 2))])
    labels = np.array([0] * 10 + [1] * 10)

    score = silhouette_on_embedding(emb, labels)

    assert score > 0.95


def test_silhouette_returns_nan_for_single_cluster():
    emb = np.zeros((10, 2))

    assert np.isnan(silhouette_on_embedding(emb, np.zeros(10, dtype=int)))


def test_silhouette_supports_sample_size():
    X, y = make_blobs(n_samples=200, centers=3, n_features=4, cluster_std=0.5, random_state=0)

    full = silhouette_on_embedding(X, y, random_state=0)
    sub = silhouette_on_embedding(X, y, sample_size=50, random_state=0)

    assert sub == pytest.approx(full, abs=0.2)


def test_modularity_higher_for_correct_partition_than_random():
    K = np.array(
        [
            [1.0, 0.9, 0.0, 0.0],
            [0.9, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.9],
            [0.0, 0.0, 0.9, 1.0],
        ]
    )

    correct = modularity_on_kernel(K, np.array([0, 0, 1, 1]))
    incorrect = modularity_on_kernel(K, np.array([0, 1, 0, 1]))

    assert correct > incorrect


def test_modularity_in_theoretical_range():
    rng = np.random.default_rng(0)
    K = rng.uniform(0.0, 1.0, size=(10, 10))
    K = (K + K.T) / 2.0

    Q = modularity_on_kernel(K, rng.integers(0, 3, size=10))

    assert -0.5 <= Q <= 1.0


def test_modularity_zero_for_empty_kernel():
    K = np.zeros((4, 4))

    assert modularity_on_kernel(K, np.array([0, 0, 1, 1])) == 0.0


def test_modularity_rejects_shape_mismatches():
    with pytest.raises(ValueError, match="square"):
        modularity_on_kernel(np.zeros((3, 4)), np.array([0, 0, 0, 1]))
    with pytest.raises(ValueError, match="length"):
        modularity_on_kernel(np.eye(3), np.array([0, 1]))
