import numpy as np
import pytest
from sklearn.datasets import make_blobs

from yggdrasil.clustering import DiscriminativeForestEmbedding
from yggdrasil.clustering.diagnostics.spectrum import LeafSpectrum, compute_leaf_spectrum
from yggdrasil.clustering.kernel import leaf_kernel
from yggdrasil.clustering.selector import (
    CandidateInputs,
    ClusterSelectionResult,
    SpectralClusterCountSelector,
)


def _spectra_for_blobs(n_centers, n_seeds, n_estimators=50, n_components=15):
    X, _ = make_blobs(
        n_samples=120, centers=n_centers, n_features=4, cluster_std=0.5, random_state=0
    )
    spectra = []
    embeddings = []
    for seed in range(n_seeds):
        emb = DiscriminativeForestEmbedding(
            n_estimators=n_estimators, sparse_output=True, random_state=seed
        ).fit(X)
        Z = emb.transform(X)
        spectra.append(
            compute_leaf_spectrum(
                Z, n_components=n_components, n_estimators=emb.n_estimators, random_state=seed
            )
        )
        embeddings.append((Z, emb.n_estimators))
    return X, spectra, embeddings


def test_selector_rejects_empty_spectra_list():
    with pytest.raises(ValueError, match="at least one"):
        SpectralClusterCountSelector().select([])


def test_eigengap_strategy_picks_three_clusters_on_three_blobs():
    _, spectra, _ = _spectra_for_blobs(n_centers=3, n_seeds=3)

    selector = SpectralClusterCountSelector(strategy="eigengap")
    result = selector.select(spectra)

    assert isinstance(result, ClusterSelectionResult)
    assert result.n_clusters == 3
    assert result.confidence == "high"
    assert result.strategy == "eigengap"


def test_eigengap_strategy_falls_back_to_two_with_low_confidence_on_single_blob():
    _, spectra, _ = _spectra_for_blobs(n_centers=1, n_seeds=3)

    result = SpectralClusterCountSelector(strategy="eigengap").select(spectra)

    assert result.n_clusters == 2
    assert result.confidence == "low"


def test_composite_strategy_requires_candidate_inputs():
    """Composite mode without `candidate_inputs` is a programmer error, not a fallback."""
    _, spectra, _ = _spectra_for_blobs(n_centers=3, n_seeds=2)

    with pytest.raises(ValueError, match="requires `candidate_inputs`"):
        SpectralClusterCountSelector().select(spectra)


def test_composite_strategy_auc_gate_runs_without_candidate_inputs():
    """The AUC gate fires before candidate scoring, so missing inputs are tolerated there."""
    _, spectra, _ = _spectra_for_blobs(n_centers=3, n_seeds=2)
    selector = SpectralClusterCountSelector(min_discriminator_auc=0.6)

    result = selector.select(spectra, discriminator_auc=0.4)

    assert result.n_clusters == 2
    assert result.gating_reason == "discriminator_auc_below_threshold"


def test_composite_oob_auc_gate_overrides_spectrum():
    _, spectra, _ = _spectra_for_blobs(n_centers=3, n_seeds=3)
    selector = SpectralClusterCountSelector(min_discriminator_auc=0.6)

    result = selector.select(spectra, discriminator_auc=0.45)

    assert result.n_clusters == 2
    assert result.confidence == "low"
    assert result.gating_reason == "discriminator_auc_below_threshold"


def test_composite_records_per_k_diagnostics_with_candidate_inputs():
    _, spectra, _ = _spectra_for_blobs(n_centers=3, n_seeds=3)
    selector = SpectralClusterCountSelector()
    candidates = selector.candidate_set(spectra)
    embeddings: dict[int, list[np.ndarray]] = {}
    labelings: dict[int, list[np.ndarray]] = {}
    rng = np.random.default_rng(0)
    for k in candidates:
        embeddings[k] = [spectrum.eigenvectors[:, : max(k, 2)] for spectrum in spectra]
        labelings[k] = [rng.integers(0, k, size=spectra[0].eigenvectors.shape[0]) for _ in spectra]
    inputs = CandidateInputs(labelings=labelings, embeddings=embeddings)

    result = selector.select(spectra, candidate_inputs=inputs, discriminator_auc=0.9)

    assert result.strategy == "composite"
    assert set(result.silhouette_per_k) == set(candidates)
    assert set(result.stability_per_k) == set(candidates)
    assert set(result.rotation_cost_per_k) == set(candidates)
    assert set(result.composite_score_per_k) == set(candidates)


def test_candidate_set_unions_eigengap_and_cumulative_mass():
    _, spectra, _ = _spectra_for_blobs(n_centers=3, n_seeds=2)

    candidates = SpectralClusterCountSelector().candidate_set(spectra)

    assert len(candidates) >= 1
    assert all(k >= 2 for k in candidates)


def test_localization_threshold_drops_high_ipr_candidates():
    """IPR threshold filters out candidates whose top eigenvectors are localized."""
    eigvals = np.array([10.0, 5.0, 4.0, 0.5, 0.4, 0.3])
    n = 30
    rng = np.random.default_rng(0)
    eigvecs = rng.normal(size=(n, eigvals.size))
    eigvecs[:, 1] = 0.0
    eigvecs[0, 1] = 1.0  # extremely localized second mode (full IPR ~ 1)
    eigvecs, _ = np.linalg.qr(eigvecs)
    spectrum = LeafSpectrum(
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        n_estimators=50,
    )

    selector = SpectralClusterCountSelector(localization_threshold=0.2)

    candidates = selector.candidate_set([spectrum])
    assert all(k != 2 for k in candidates) or len(candidates) == 0


def test_eigengap_strategy_bit_identical_to_v1_modal_majority():
    """With strategy='eigengap', the result reproduces the v1 logic exactly."""
    _, spectra, _ = _spectra_for_blobs(n_centers=3, n_seeds=3)

    selector = SpectralClusterCountSelector(strategy="eigengap")
    result = selector.select(spectra)

    assert tuple(result.proposed_k_per_seed.tolist()) == (3, 3, 3)
    assert result.strategy == "eigengap"


def test_explicit_kernel_modularity_path_runs():
    _, spectra, embeddings = _spectra_for_blobs(n_centers=3, n_seeds=2)
    selector = SpectralClusterCountSelector(modularity_weight=1.0)
    candidates = selector.candidate_set(spectra)
    embs: dict[int, list[np.ndarray]] = {}
    lbls: dict[int, list[np.ndarray]] = {}
    rng = np.random.default_rng(0)
    for k in candidates:
        embs[k] = [s.eigenvectors[:, : max(k, 2)] for s in spectra]
        lbls[k] = [rng.integers(0, k, size=spectra[0].eigenvectors.shape[0]) for _ in spectra]
    Ks = [leaf_kernel(Z, n_estimators=n_est) for Z, n_est in embeddings]
    inputs = CandidateInputs(labelings=lbls, embeddings=embs, kernels=Ks)

    result = selector.select(spectra, candidate_inputs=inputs, discriminator_auc=0.9)

    assert set(result.modularity_per_k) == set(candidates)
    assert all(np.isfinite(v) for v in result.modularity_per_k.values())
