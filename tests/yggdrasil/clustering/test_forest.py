import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.utils._testing import assert_allclose
from sklearn.utils.estimator_checks import parametrize_with_checks

from yggdrasil.clustering import DiscriminativeForestEmbedding


def _expected_failed_checks(estimator):
    reason = (
        "Duplicating a real row also changes the empirical marginals from "
        "which synthetic rows are drawn, so weight equivalence to row "
        "repetition cannot hold for this estimator."
    )
    return {
        "check_sample_weight_equivalence_on_dense_data": reason,
        "check_sample_weight_equivalence_on_sparse_data": reason,
    }


@parametrize_with_checks(
    [
        DiscriminativeForestEmbedding(n_estimators=5, random_state=0),
        DiscriminativeForestEmbedding(n_estimators=5, backend="extra_trees", random_state=0),
    ],
    expected_failed_checks=_expected_failed_checks,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


@pytest.fixture
def two_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=0.5, size=(40, 4))
    b = rng.normal(loc=5.0, scale=0.5, size=(40, 4))
    return np.vstack([a, b])


@pytest.mark.parametrize("backend", ["random_forest", "extra_trees"])
def test_fit_transform_returns_sparse_one_hot_encoding(backend, two_blobs):
    est = DiscriminativeForestEmbedding(
        n_estimators=8, backend=backend, sparse_output=True, random_state=0
    )

    embedding = est.fit_transform(two_blobs)

    assert sp.issparse(embedding)
    assert embedding.shape[0] == two_blobs.shape[0]
    assert (embedding.sum(axis=1) == est.n_estimators).all()


@pytest.mark.parametrize("backend", ["random_forest", "extra_trees"])
def test_fit_transform_returns_dense_leaf_indices(backend, two_blobs):
    est = DiscriminativeForestEmbedding(
        n_estimators=8, backend=backend, sparse_output=False, random_state=0
    )

    embedding = est.fit_transform(two_blobs)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (two_blobs.shape[0], est.n_estimators)
    assert np.issubdtype(embedding.dtype, np.integer)


@pytest.mark.parametrize(
    ("backend", "expected_cls"),
    [("random_forest", RandomForestClassifier), ("extra_trees", ExtraTreesClassifier)],
)
def test_backend_selects_expected_forest_class(backend, expected_cls, two_blobs):
    est = DiscriminativeForestEmbedding(n_estimators=4, backend=backend, random_state=0).fit(
        two_blobs
    )

    assert isinstance(est.forest_, expected_cls)
    assert len(est.forest_.estimators_) == est.n_estimators


def test_unknown_backend_raises_value_error(two_blobs):
    est = DiscriminativeForestEmbedding(n_estimators=4, backend="not_a_backend")

    with pytest.raises(ValueError, match="backend must be one of"):
        est.fit(two_blobs)


@pytest.mark.parametrize("backend", ["random_forest", "extra_trees"])
def test_transform_is_deterministic_given_seed(backend, two_blobs):
    a = DiscriminativeForestEmbedding(
        n_estimators=6, backend=backend, sparse_output=False, random_state=42
    ).fit_transform(two_blobs)
    b = DiscriminativeForestEmbedding(
        n_estimators=6, backend=backend, sparse_output=False, random_state=42
    ).fit_transform(two_blobs)

    assert_allclose(a, b)


@pytest.mark.parametrize("backend", ["random_forest", "extra_trees"])
def test_transform_matches_fit_transform_on_training_data(backend, two_blobs):
    est = DiscriminativeForestEmbedding(
        n_estimators=6, backend=backend, sparse_output=False, random_state=0
    )

    embedded_via_fit_transform = est.fit_transform(two_blobs)
    embedded_via_transform = est.transform(two_blobs)

    assert_allclose(embedded_via_fit_transform, embedded_via_transform)


def test_extra_trees_yields_different_embedding_than_random_forest(two_blobs):
    rf = DiscriminativeForestEmbedding(
        n_estimators=8,
        backend="random_forest",
        sparse_output=False,
        random_state=0,
    ).fit_transform(two_blobs)
    et = DiscriminativeForestEmbedding(
        n_estimators=8,
        backend="extra_trees",
        sparse_output=False,
        random_state=0,
    ).fit_transform(two_blobs)

    assert rf.shape == et.shape
    assert not np.array_equal(rf, et)
