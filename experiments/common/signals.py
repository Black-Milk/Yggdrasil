"""Per-(k, reseed) signal evaluation for composite-scorer experiments.

The composite cluster-count selector in
:class:`yggdrasil.clustering.SpectralClusterCountSelector` consumes a
handful of independently-defined signals (silhouette, label stability,
rotation cost, eigengap support, optionally Newman modularity), z-scores
each across the candidate set, and picks the argmax. This module
evaluates each signal **independently** on a user-specified ``k`` grid
so we can study its individual bias and variance pattern across
datasets, instead of only seeing the post-composite verdict.

The two main entry points are:

- :func:`evaluate_signals` — compute every signal on every ``(k, reseed)``
  pair, return a dataframe-friendly long-form list of records.
- :func:`v2_verdict` — run the production
  :class:`~yggdrasil.clustering.DiscriminativeForestClusterer` once and
  report what it picked, its confidence, and ARI to ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from yggdrasil.clustering import DiscriminativeForestClusterer, DiscriminativeForestEmbedding
from yggdrasil.clustering.diagnostics import (
    compute_leaf_spectrum,
    cumulative_spectral_mass,
    discriminator_oob_auc,
    effective_rank,
    label_stability,
    rotation_cost,
)


@dataclass(frozen=True)
class SignalRecord:
    """One row of the long-form signal-evaluation table."""

    dataset: str
    k_true: int
    n_samples: int
    k: int
    silhouette_mean: float
    silhouette_std: float
    rotation_cost_mean: float
    rotation_cost_std: float
    label_stability: float
    eigengap_support: float
    ari_kmeans_to_truth_mean: float
    ari_kmeans_to_truth_std: float
    effective_rank_mean: float
    cum_mass_k_at_0_9_mean: float
    discriminator_oob_auc_mean: float
    n_reseeds: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "k_true": self.k_true,
            "n_samples": self.n_samples,
            "k": self.k,
            "silhouette_mean": self.silhouette_mean,
            "silhouette_std": self.silhouette_std,
            "rotation_cost_mean": self.rotation_cost_mean,
            "rotation_cost_std": self.rotation_cost_std,
            "label_stability": self.label_stability,
            "eigengap_support": self.eigengap_support,
            "ari_kmeans_to_truth_mean": self.ari_kmeans_to_truth_mean,
            "ari_kmeans_to_truth_std": self.ari_kmeans_to_truth_std,
            "effective_rank_mean": self.effective_rank_mean,
            "cum_mass_k_at_0_9_mean": self.cum_mass_k_at_0_9_mean,
            "discriminator_oob_auc_mean": self.discriminator_oob_auc_mean,
            "n_reseeds": self.n_reseeds,
        }


def _row_normalize(U: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    return U / np.where(norms > 0, norms, 1.0)


def _spectral_embedding(eigenvectors: np.ndarray, k: int, drop_leading_mode: bool) -> np.ndarray:
    n_components = eigenvectors.shape[1]
    start = 1 if (drop_leading_mode and n_components > k) else 0
    end = min(start + k, n_components)
    if end - start < 1:
        start = 0
        end = min(k, n_components)
    return _row_normalize(eigenvectors[:, start:end])


def _propose_k_from_eigengaps(
    eigenvalues: np.ndarray, *, drop_leading_mode: bool, max_k: int
) -> int:
    """Mirror SpectralClusterCountSelector._propose_from_one for parity."""
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    offset = 1 if (drop_leading_mode and eigs.size > 1) else 0
    eigs_eff = eigs[offset:]
    if eigs_eff.size < 2:
        return 2
    min_k_eff = max(1, 2 - offset)
    max_k_eff = int(min(max_k - offset, eigs_eff.size - 1))
    if max_k_eff < min_k_eff:
        return 2
    candidate_k_eff = np.arange(min_k_eff, max_k_eff + 1)
    num = eigs_eff[candidate_k_eff - 1] - eigs_eff[candidate_k_eff]
    denom = eigs_eff[candidate_k_eff] + 1e-12
    gaps = num / denom
    return int(candidate_k_eff[int(np.argmax(gaps))]) + offset


def evaluate_signals(
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    dataset_name: str,
    k_grid: tuple[int, ...] = (2, 3, 4, 5, 6, 7),
    n_estimators: int = 100,
    n_reseeds: int = 3,
    drop_leading_mode: bool = True,
    n_components_buffer: int = 6,
    max_k_for_eigengap: int = 20,
    random_state: int = 0,
) -> list[SignalRecord]:
    """Compute every diagnostic signal on every ``(k, reseed)`` pair.

    Returns one :class:`SignalRecord` per ``k`` in ``k_grid``.
    """
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_reseeds)

    embeddings_per_seed: list[np.ndarray] = []
    eigenvalues_per_seed: list[np.ndarray] = []
    proposed_k_per_seed: list[int] = []
    effective_rank_per_seed: list[float] = []
    cum_mass_per_seed: list[int] = []
    auc_per_seed: list[float] = []

    n_components = max(k_grid) + n_components_buffer
    if drop_leading_mode:
        n_components += 1

    for seed in seeds:
        emb = DiscriminativeForestEmbedding(
            n_estimators=n_estimators, sparse_output=True, oob_score=True, random_state=int(seed)
        )
        Z = emb.fit_transform(X)
        spectrum = compute_leaf_spectrum(
            Z,
            n_components=min(n_components, *Z.shape),
            n_estimators=n_estimators,
            random_state=int(seed),
        )
        embeddings_per_seed.append(spectrum.eigenvectors)
        eigenvalues_per_seed.append(spectrum.eigenvalues)
        proposed_k_per_seed.append(
            _propose_k_from_eigengaps(
                spectrum.eigenvalues,
                drop_leading_mode=drop_leading_mode,
                max_k=max_k_for_eigengap,
            )
        )
        effective_rank_per_seed.append(effective_rank(spectrum.eigenvalues))
        cum_mass_per_seed.append(cumulative_spectral_mass(spectrum, threshold=0.9))
        try:
            auc_per_seed.append(float(discriminator_oob_auc(emb.forest_, emb.y_disc_)))
        except AttributeError, ValueError:
            auc_per_seed.append(float("nan"))

    proposed_arr = np.asarray(proposed_k_per_seed, dtype=np.int64)
    records: list[SignalRecord] = []
    for k in k_grid:
        per_seed_silhouettes: list[float] = []
        per_seed_rotations: list[float] = []
        per_seed_aris: list[float] = []
        labelings: list[np.ndarray] = []
        for seed, eigvecs in zip(seeds, embeddings_per_seed, strict=True):
            U = _spectral_embedding(eigvecs, k, drop_leading_mode)
            labels = KMeans(
                n_clusters=k,
                n_init=10,
                random_state=int(seed),
            ).fit_predict(U)
            labelings.append(labels)
            sil = float(silhouette_score(U, labels)) if len(np.unique(labels)) > 1 else float("nan")
            per_seed_silhouettes.append(sil)
            per_seed_rotations.append(rotation_cost(U, n_restarts=5, random_state=int(seed)))
            per_seed_aris.append(float(adjusted_rand_score(y_true, labels)))

        stab = (
            float(label_stability(labelings, metric="ari")) if len(labelings) >= 2 else float("nan")
        )
        eigengap_supp = float(np.mean(proposed_arr == k))
        records.append(
            SignalRecord(
                dataset=dataset_name,
                k_true=int(np.unique(y_true).size),
                n_samples=int(X.shape[0]),
                k=int(k),
                silhouette_mean=float(np.nanmean(per_seed_silhouettes)),
                silhouette_std=float(np.nanstd(per_seed_silhouettes)),
                rotation_cost_mean=float(np.nanmean(per_seed_rotations)),
                rotation_cost_std=float(np.nanstd(per_seed_rotations)),
                label_stability=stab,
                eigengap_support=eigengap_supp,
                ari_kmeans_to_truth_mean=float(np.nanmean(per_seed_aris)),
                ari_kmeans_to_truth_std=float(np.nanstd(per_seed_aris)),
                effective_rank_mean=float(np.mean(effective_rank_per_seed)),
                cum_mass_k_at_0_9_mean=float(np.mean(cum_mass_per_seed)),
                discriminator_oob_auc_mean=float(np.nanmean(auc_per_seed)),
                n_reseeds=int(n_reseeds),
            )
        )
    return records


@dataclass(frozen=True)
class V2Verdict:
    """Summary of what ``DiscriminativeForestClusterer`` decides on a dataset."""

    dataset: str
    n_clusters_picked: int
    confidence: str
    gating_reason: str | None
    proposed_k_per_seed: list[int]
    discriminator_auc: float | None
    composite_score_per_k: dict[int, float]
    silhouette_per_k: dict[int, float]
    stability_per_k: dict[int, float]
    rotation_cost_per_k: dict[int, float]
    ari_to_truth: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_clusters_picked": self.n_clusters_picked,
            "confidence": self.confidence,
            "gating_reason": self.gating_reason,
            "proposed_k_per_seed": self.proposed_k_per_seed,
            "discriminator_auc": self.discriminator_auc,
            "composite_score_per_k": {
                int(k): float(v) for k, v in self.composite_score_per_k.items()
            },
            "silhouette_per_k": {int(k): float(v) for k, v in self.silhouette_per_k.items()},
            "stability_per_k": {int(k): float(v) for k, v in self.stability_per_k.items()},
            "rotation_cost_per_k": {int(k): float(v) for k, v in self.rotation_cost_per_k.items()},
            "ari_to_truth": self.ari_to_truth,
        }


def v2_verdict(
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    dataset_name: str,
    n_estimators: int = 100,
    n_selection_resamples: int = 3,
    random_state: int = 0,
    cluster_selection: str = "composite",
) -> V2Verdict:
    """Run the production v2 estimator on a dataset and report its verdict."""
    est = DiscriminativeForestClusterer(
        n_estimators=n_estimators,
        n_selection_resamples=n_selection_resamples,
        cluster_selection=cluster_selection,
        random_state=random_state,
    ).fit(X)
    cs = est.cluster_selection_
    return V2Verdict(
        dataset=dataset_name,
        n_clusters_picked=int(est.n_clusters_),
        confidence=str(cs.confidence),
        gating_reason=cs.gating_reason,
        proposed_k_per_seed=[int(v) for v in cs.proposed_k_per_seed.tolist()],
        discriminator_auc=(None if cs.discriminator_auc is None else float(cs.discriminator_auc)),
        composite_score_per_k=dict(cs.composite_score_per_k),
        silhouette_per_k=dict(cs.silhouette_per_k),
        stability_per_k=dict(cs.stability_per_k),
        rotation_cost_per_k=dict(cs.rotation_cost_per_k),
        ari_to_truth=float(adjusted_rand_score(y_true, est.labels_)),
    )


def best_k_per_signal(
    records: list[SignalRecord],
    *,
    higher_is_better: dict[str, bool] | None = None,
) -> dict[str, int]:
    """For each signal, return the ``k`` that "wins" according to that signal alone.

    For the default-equipped signals this answers the question:
        "If we let *only* this signal vote, what k would it pick?"
    """
    if higher_is_better is None:
        higher_is_better = {
            "silhouette_mean": True,
            "rotation_cost_mean": False,
            "label_stability": True,
            "eigengap_support": True,
            "ari_kmeans_to_truth_mean": True,
        }
    rows = [r.as_dict() for r in records]
    out: dict[str, int] = {}
    for signal, hib in higher_is_better.items():
        scored = [(row[signal], int(row["k"])) for row in rows if np.isfinite(row[signal])]
        if not scored:
            continue
        sign = 1 if hib else -1
        best = max(scored, key=lambda sk: sign * sk[0])
        out[signal] = int(best[1])
    return out
