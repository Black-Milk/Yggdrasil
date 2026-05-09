import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_blobs
from sklearn.utils._testing import assert_allclose

from yggdrasil.clustering import DiscriminativeForestEmbedding
from yggdrasil.clustering.diagnostics.spectrum import (
    LeafSpectrum,
    compute_leaf_spectrum,
    cumulative_spectral_mass,
    effective_rank,
    eigengap_curve,
    inverse_participation_ratios,
)
from yggdrasil.clustering.kernel import leaf_indicator_matrix, leaf_kernel


@pytest.fixture
def three_blob_embedding():
    X, _ = make_blobs(n_samples=120, centers=3, n_features=4, cluster_std=0.5, random_state=0)
    embedding = DiscriminativeForestEmbedding(
        n_estimators=30, sparse_output=True, random_state=0
    ).fit(X)
    Z = embedding.transform(X)
    return X, Z, embedding.n_estimators


def test_compute_leaf_spectrum_top_eigenvalues_match_dense_kernel(three_blob_embedding):
    _, Z, n_estimators = three_blob_embedding

    spectrum = compute_leaf_spectrum(Z, n_components=10, n_estimators=n_estimators, random_state=0)

    K = leaf_kernel(Z, n_estimators=n_estimators)
    dense_eigs = np.linalg.eigvalsh(K)[::-1]

    assert isinstance(spectrum, LeafSpectrum)
    assert spectrum.eigenvalues.shape == (10,)
    assert spectrum.eigenvectors.shape == (Z.shape[0], 10)
    assert_allclose(spectrum.eigenvalues, dense_eigs[:10], atol=1e-8)
    assert (np.diff(spectrum.eigenvalues) <= 1e-8).all()


def test_compute_leaf_spectrum_caps_components_at_min_dim():
    Z = leaf_indicator_matrix(np.array([[0, 0], [1, 1], [0, 0]]))

    spectrum = compute_leaf_spectrum(Z, n_components=99, n_estimators=2)

    assert spectrum.eigenvalues.shape[0] <= min(Z.shape)
    assert spectrum.eigenvectors.shape[0] == Z.shape[0]


def test_compute_leaf_spectrum_rejects_empty_input():
    Z = sp.csr_matrix(np.zeros((0, 3)))

    with pytest.raises(ValueError, match="non-empty"):
        compute_leaf_spectrum(Z, n_components=2, n_estimators=2)


def test_effective_rank_matches_hand_computed_entropy():
    eigs = np.array([4.0, 1.0, 1.0])

    r = effective_rank(eigs)

    p = eigs / eigs.sum()
    expected = float(np.exp(-np.sum(p * np.log(p))))
    assert r == pytest.approx(expected)


def test_effective_rank_handles_all_zero_input():
    assert effective_rank(np.zeros(5)) == 0.0


def test_inverse_participation_ratios_matches_known_vectors():
    V = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    ipr = inverse_participation_ratios(V)

    expected = np.array([4.0 * (0.5**4), 1.0, 0.0])
    assert_allclose(ipr, expected, atol=1e-12)


def test_inverse_participation_ratios_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D array"):
        inverse_participation_ratios(np.array([1.0, 0.0, 0.0]))


def test_cumulative_spectral_mass_picks_smallest_k_at_threshold():
    eigs = np.array([4.0, 1.0, 1.0])

    assert cumulative_spectral_mass(eigs, threshold=0.5) == 1
    assert cumulative_spectral_mass(eigs, threshold=0.7) == 2
    assert cumulative_spectral_mass(eigs, threshold=0.95) == 3


def test_cumulative_spectral_mass_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        cumulative_spectral_mass(np.array([1.0, 1.0]), threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        cumulative_spectral_mass(np.array([1.0, 1.0]), threshold=1.5)


def test_cumulative_spectral_mass_accepts_leaf_spectrum_object(three_blob_embedding):
    _, Z, n_estimators = three_blob_embedding
    spectrum = compute_leaf_spectrum(Z, n_components=10, n_estimators=n_estimators, random_state=0)

    k = cumulative_spectral_mass(spectrum, threshold=0.9)

    assert 1 <= k <= spectrum.eigenvalues.size


def test_eigengap_curve_returns_n_minus_one_relative_gaps():
    eigs = np.array([4.0, 1.0, 0.5])

    gaps = eigengap_curve(eigs)

    assert gaps.shape == (2,)
    assert gaps[0] == pytest.approx(3.0, rel=1e-6)
    assert gaps[1] == pytest.approx(1.0, rel=1e-6)


def test_eigengap_curve_returns_empty_for_single_eigenvalue():
    assert eigengap_curve(np.array([1.0])).shape == (0,)
