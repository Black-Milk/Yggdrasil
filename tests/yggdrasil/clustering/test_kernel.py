import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_blobs
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._testing import assert_allclose

from yggdrasil.clustering.kernel import (
    kernel_from_leaves,
    leaf_dissimilarity,
    leaf_indicator_matrix,
    leaf_kernel,
)


@pytest.fixture
def leaves_small():
    return np.array([[0, 1, 0], [0, 2, 1], [1, 1, 0], [1, 2, 1]])


@pytest.fixture
def leaves_make_blobs():
    X, _ = make_blobs(n_samples=30, centers=3, n_features=4, cluster_std=0.5, random_state=0)
    rng = np.random.default_rng(0)
    n_estimators = 5
    leaves = rng.integers(low=0, high=4, size=(X.shape[0], n_estimators))
    return leaves


def test_leaf_indicator_matrix_returns_sparse_one_hot(leaves_small):
    Z = leaf_indicator_matrix(leaves_small)

    assert sp.issparse(Z)
    assert Z.shape == (4, 6)
    assert (Z.sum(axis=1) == leaves_small.shape[1]).all()


def test_leaf_indicator_matrix_matches_one_hot_encoder(leaves_make_blobs):
    leaves = leaves_make_blobs

    Z = leaf_indicator_matrix(leaves)
    expected = OneHotEncoder(sparse_output=True, dtype=np.float64).fit_transform(leaves)

    assert Z.shape == expected.shape
    assert_allclose(Z.toarray(), expected.toarray())


def test_leaf_indicator_matrix_rejects_1d():
    with pytest.raises(ValueError, match="2-D array"):
        leaf_indicator_matrix(np.array([0, 1, 2]))


def test_leaf_kernel_is_symmetric_with_unit_diagonal(leaves_small):
    Z = leaf_indicator_matrix(leaves_small)

    K = leaf_kernel(Z, n_estimators=leaves_small.shape[1])

    assert K.shape == (leaves_small.shape[0], leaves_small.shape[0])
    assert_allclose(K, K.T, atol=1e-12)
    assert_allclose(np.diag(K), np.ones(leaves_small.shape[0]))
    assert (K >= 0).all() and (K <= 1).all()


def test_leaf_kernel_infers_n_estimators_from_constant_row_sums(leaves_small):
    Z = leaf_indicator_matrix(leaves_small)

    K_inferred = leaf_kernel(Z)
    K_explicit = leaf_kernel(Z, n_estimators=leaves_small.shape[1])

    assert_allclose(K_inferred, K_explicit)


def test_leaf_kernel_rejects_non_constant_row_sums():
    Z = sp.csr_matrix(np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))

    with pytest.raises(ValueError, match="row sums"):
        leaf_kernel(Z)


def test_leaf_dissimilarity_complements_kernel(leaves_small):
    Z = leaf_indicator_matrix(leaves_small)
    K = leaf_kernel(Z, n_estimators=leaves_small.shape[1])

    D = leaf_dissimilarity(K)

    assert_allclose(D, 1.0 - K)
    assert_allclose(np.diag(D), np.zeros(leaves_small.shape[0]), atol=1e-12)


def test_kernel_from_leaves_matches_two_step_pipeline(leaves_make_blobs):
    leaves = leaves_make_blobs

    K_direct = kernel_from_leaves(leaves)
    Z = leaf_indicator_matrix(leaves)
    K_indirect = leaf_kernel(Z, n_estimators=leaves.shape[1])

    assert_allclose(K_direct, K_indirect)
