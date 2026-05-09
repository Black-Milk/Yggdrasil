import numpy as np
import pytest

from yggdrasil.utils.synthetic import generate_synthetic_features, permutation_sample_column


def test_permutation_sample_column_preserves_values_without_replacement():
    X = np.arange(10)

    sampled = permutation_sample_column(X, random_state=0)

    assert sampled.shape == X.shape
    assert np.array_equal(np.sort(sampled), X)
    assert not np.array_equal(sampled, X)


def test_permutation_sample_column_can_draw_subset_without_replacement():
    X = np.arange(10)

    sampled = permutation_sample_column(X, n_samples=4, random_state=0)

    assert sampled.shape == (4,)
    assert len(np.unique(sampled)) == sampled.shape[0]
    assert np.all(np.isin(sampled, X))


def test_permutation_sample_column_rejects_oversampling():
    X = np.arange(3)

    with pytest.raises(ValueError, match="n_samples cannot exceed"):
        permutation_sample_column(X, n_samples=4, random_state=0)


def test_generate_synthetic_features_permutation_preserves_column_marginals():
    X = np.array(
        [
            [0, 10],
            [1, 11],
            [2, 12],
            [3, 13],
            [4, 14],
            [5, 15],
        ]
    )

    sampled = generate_synthetic_features(X, method="permutation", random_state=0)

    assert sampled.shape == X.shape
    for column in range(X.shape[1]):
        assert np.array_equal(np.sort(sampled[:, column]), X[:, column])
    assert not np.array_equal(sampled, X)
