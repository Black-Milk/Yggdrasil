import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils._testing import assert_allclose
from sklearn.utils.estimator_checks import parametrize_with_checks

from yggdrasil.clustering import DiscriminativeForestClusterer
from yggdrasil.clustering.diagnostics import ClusterSelectionResult


@parametrize_with_checks(
    [
        DiscriminativeForestClusterer(
            n_clusters=2,
            n_estimators=5,
            n_selection_resamples=1,
            random_state=0,
        ),
        DiscriminativeForestClusterer(
            n_estimators=5,
            n_selection_resamples=1,
            random_state=0,
        ),
        DiscriminativeForestClusterer(
            n_estimators=5,
            n_selection_resamples=1,
            cluster_selection="eigengap",
            random_state=0,
        ),
    ]
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


@pytest.fixture
def three_blobs():
    X, y = make_blobs(n_samples=120, centers=3, n_features=4, cluster_std=0.5, random_state=0)
    return X, y


@pytest.fixture
def four_blobs():
    X, y = make_blobs(n_samples=160, centers=4, n_features=4, cluster_std=0.5, random_state=0)
    return X, y


@pytest.fixture
def single_blob():
    X, _ = make_blobs(n_samples=80, centers=1, n_features=4, cluster_std=1.0, random_state=0)
    return X


def test_fit_predict_returns_labels_with_correct_shape(three_blobs):
    X, _ = three_blobs

    est = DiscriminativeForestClusterer(n_estimators=30, n_selection_resamples=1, random_state=0)
    labels = est.fit_predict(X)

    assert labels.shape == (X.shape[0],)
    assert labels is est.labels_
    assert np.issubdtype(labels.dtype, np.integer)


def test_auto_n_clusters_recovers_three_blobs(three_blobs):
    X, y = three_blobs

    est = DiscriminativeForestClusterer(
        n_estimators=50, n_selection_resamples=3, random_state=0
    ).fit(X)

    assert est.n_clusters_ == 3
    assert est.cluster_selection_.confidence == "high"
    assert adjusted_rand_score(y, est.labels_) > 0.9


def test_auto_n_clusters_recovers_four_blobs(four_blobs):
    X, y = four_blobs

    est = DiscriminativeForestClusterer(
        n_estimators=80, n_selection_resamples=3, random_state=0
    ).fit(X)

    assert est.n_clusters_ == 4
    assert est.cluster_selection_.confidence == "high"
    assert adjusted_rand_score(y, est.labels_) > 0.9


def test_eigengap_strategy_still_recovers_three_blobs(three_blobs):
    X, y = three_blobs

    est = DiscriminativeForestClusterer(
        n_estimators=50,
        n_selection_resamples=3,
        cluster_selection="eigengap",
        random_state=0,
    ).fit(X)

    assert est.n_clusters_ == 3
    assert est.cluster_selection_.strategy == "eigengap"
    assert adjusted_rand_score(y, est.labels_) > 0.9


def test_explicit_n_clusters_bypasses_auto_selection(three_blobs):
    X, _ = three_blobs

    est = DiscriminativeForestClusterer(
        n_clusters=4,
        n_estimators=20,
        n_selection_resamples=1,
        random_state=0,
    ).fit(X)

    assert est.n_clusters_ == 4
    assert len(np.unique(est.labels_)) == 4
    assert est.cluster_selection_.strategy == "explicit"
    assert est.cluster_selection_.confidence == "high"


def test_no_structure_falls_back_via_auc_gate(single_blob):
    X = single_blob

    est = DiscriminativeForestClusterer(
        n_estimators=50, n_selection_resamples=3, random_state=0
    ).fit(X)

    assert est.n_clusters_ == 2
    assert est.cluster_selection_.confidence == "low"
    assert est.cluster_selection_.gating_reason == "discriminator_auc_below_threshold"
    assert est.labels_.shape == (X.shape[0],)


def test_cluster_selection_result_is_populated(three_blobs):
    X, _ = three_blobs

    est = DiscriminativeForestClusterer(
        n_estimators=50, n_selection_resamples=2, random_state=0
    ).fit(X)

    result = est.cluster_selection_
    assert isinstance(result, ClusterSelectionResult)
    assert result.eigenvalues.ndim == 1
    assert result.eigenvalues.size >= 2
    assert result.eigengaps.shape == (result.eigenvalues.size - 1,)
    assert result.localization.shape == result.eigenvalues.shape
    assert result.proposed_k_per_seed.shape == (2,)
    assert result.effective_rank > 0.0
    assert result.discriminator_auc is not None
    assert len(result.silhouette_per_k) >= 1
    assert len(result.stability_per_k) >= 1
    assert len(result.rotation_cost_per_k) >= 1
    assert len(result.composite_score_per_k) >= 1


def test_modularity_weight_enables_kernel_path(three_blobs):
    X, _ = three_blobs

    est = DiscriminativeForestClusterer(
        n_estimators=30,
        n_selection_resamples=2,
        modularity_weight=1.0,
        random_state=0,
    ).fit(X)

    modularity = est.cluster_selection_.modularity_per_k
    assert len(modularity) >= 1
    assert all(np.isfinite(v) for v in modularity.values())
    # Newman modularity on a clustered proximity kernel must be strictly
    # positive at the winning k. The previous truncated `U Λ Uᵀ`
    # reconstruction routinely violated this (negative off-diagonals,
    # under-counted degrees); leaf_kernel(Z) restores it.
    assert modularity[est.n_clusters_] > 0.0


def test_fit_is_deterministic_under_fixed_random_state(three_blobs):
    X, _ = three_blobs
    kwargs = {"n_estimators": 20, "n_selection_resamples": 1, "random_state": 7}

    a = DiscriminativeForestClusterer(**kwargs).fit(X)
    b = DiscriminativeForestClusterer(**kwargs).fit(X)

    assert a.n_clusters_ == b.n_clusters_
    assert_allclose(a.spectral_embedding_, b.spectral_embedding_)
    assert np.array_equal(a.labels_, b.labels_)


def test_extra_trees_backend_runs(three_blobs):
    X, _ = three_blobs

    est = DiscriminativeForestClusterer(
        n_clusters=3,
        backend="extra_trees",
        n_estimators=20,
        n_selection_resamples=1,
        random_state=0,
    ).fit(X)

    assert est.n_clusters_ == 3
    assert est.labels_.shape == (X.shape[0],)
