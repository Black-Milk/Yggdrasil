"""Spectral diagnostics and cluster-count selection for forest leaf kernels.

This module provides the spectral pieces that sit between the sparse leaf
embedding (see :mod:`yggdrasil.clustering.kernel`) and a downstream
clustering algorithm:

- :func:`compute_leaf_spectrum` extracts the leading eigenvalues and
  eigenvectors of the leaf kernel ``K = Z Zᵀ / T`` once, via truncated
  SVD on the sparse leaf-indicator matrix ``Z``. The dense ``K`` is
  never materialized.
- :func:`effective_rank` and :func:`inverse_participation_ratios`
  summarize spectral mass and eigenvector concentration.
- :class:`SpectralClusterCountSelector` proposes ``n_clusters`` from one
  or more spectra using a defensive relative-eigengap rule, with a
  modal-:math:`k` vote across re-fitted forest seeds.

The cluster-count selector is intentionally narrow in this version:
eigengap proposes, modal-:math:`k` across reseeded forests sanity-checks,
and the result falls back to ``k = 2`` when no clear gap is present.
Localization and effective rank are computed and reported but do **not**
change the chosen ``k``; they are post-hoc warning signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence
from sklearn.utils import check_random_state

__all__ = [
    "ClusterSelectionResult",
    "LeafSpectrum",
    "SpectralClusterCountSelector",
    "compute_leaf_spectrum",
    "effective_rank",
    "inverse_participation_ratios",
]


@dataclass(frozen=True)
class LeafSpectrum:
    """Top-eigenvalue spectrum of the leaf kernel ``K = Z Zᵀ / T``.

    Computed via truncated SVD of ``Z / sqrt(n_estimators)``. The
    eigenvalues of ``K`` equal the squared singular values, and the
    eigenvectors of ``K`` equal the left singular vectors of the scaled
    ``Z``.

    Parameters
    ----------
    eigenvalues : ndarray of shape (n_components,)
        Leading eigenvalues of ``K``, in descending order.
    eigenvectors : ndarray of shape (n_samples, n_components)
        Corresponding eigenvectors of ``K``, with columns in the same
        order as ``eigenvalues``.
    n_estimators : int
        Number of trees the kernel was normalized by.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    n_estimators: int


@dataclass(frozen=True)
class ClusterSelectionResult:
    """Inspectable summary of how ``n_clusters`` was chosen.

    Parameters
    ----------
    n_clusters : int
        The chosen number of clusters.
    eigenvalues : ndarray of shape (n_components,)
        Eigenvalues of the primary leaf kernel, descending.
    eigengaps : ndarray of shape (n_components - 1,)
        Relative eigengaps ``(λ_i - λ_{i+1}) / (λ_{i+1} + ε)`` over the
        full computed spectrum.
    effective_rank : float
        Spectral entropy effective rank, ``exp(H)`` with ``H`` the
        Shannon entropy of normalized eigenvalues.
    localization : ndarray of shape (n_components,)
        Inverse participation ratios of each eigenvector; small values
        indicate a broadly distributed mode, large values indicate a
        mode concentrated on few samples.
    proposed_k_per_seed : ndarray of shape (n_seeds,)
        The eigengap-proposed ``k`` from each spectrum that was passed
        to the selector.
    strategy : str
        Selection strategy used (e.g. ``"eigengap"`` or ``"explicit"``).
    confidence : {"high", "low"}
        ``"low"`` when the maximum relative eigengap fell below the
        selector's threshold and the result fell back to ``k = 2``;
        otherwise ``"high"``.
    """

    n_clusters: int
    eigenvalues: np.ndarray
    eigengaps: np.ndarray
    effective_rank: float
    localization: np.ndarray
    proposed_k_per_seed: np.ndarray
    strategy: str
    confidence: Literal["high", "low"]


def compute_leaf_spectrum(
    Z: sp.spmatrix | np.ndarray,
    n_components: int,
    *,
    n_estimators: int | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> LeafSpectrum:
    """Compute the leading spectrum of ``K = Z Zᵀ / T`` via SVD on ``Z``.

    Uses :func:`scipy.sparse.linalg.svds` for moderately-sized inputs
    and falls back to a dense :func:`numpy.linalg.svd` for small or
    nearly-square problems where the iterative path is unreliable.

    Parameters
    ----------
    Z : {array-like, sparse matrix} of shape (n_samples, n_total_leaves)
        Sparse leaf-indicator matrix produced by
        :func:`yggdrasil.clustering.kernel.leaf_indicator_matrix` or by
        :class:`~yggdrasil.clustering.DiscriminativeForestEmbedding`.
    n_components : int
        Number of top eigenvalues to retain. Capped at
        ``min(n_samples, n_total_leaves)``.
    n_estimators : int, optional
        Number of trees in the forest. If ``None``, inferred from the
        row sums of ``Z``, which must be constant.
    random_state : int, RandomState instance or None, default=None
        Controls the random initialization of the iterative SVD. See
        :term:`Glossary <random_state>`.

    Returns
    -------
    spectrum : LeafSpectrum
        Eigenvalues, eigenvectors, and the kernel normalization constant.

    Raises
    ------
    ValueError
        If ``n_components`` is non-positive, if ``Z`` is empty, or if
        ``n_estimators`` cannot be inferred.
    """
    if not sp.issparse(Z):
        Z = sp.csr_matrix(Z)

    n_samples, n_features = Z.shape
    if n_samples == 0 or n_features == 0:
        raise ValueError("Z must be non-empty.")

    if n_estimators is None:
        row_sums = np.asarray(Z.sum(axis=1)).ravel()
        unique = np.unique(row_sums)
        if unique.size != 1:
            raise ValueError(
                "n_estimators is None but Z row sums are not constant; "
                "pass n_estimators explicitly."
            )
        n_estimators = int(unique[0])

    if n_estimators <= 0:
        raise ValueError(f"n_estimators must be a positive integer; got {n_estimators}.")

    max_components = min(n_samples, n_features)
    n_components = int(min(n_components, max_components))
    if n_components < 1:
        raise ValueError(f"n_components must be at least 1; got {n_components}.")

    scale = 1.0 / np.sqrt(n_estimators)
    rs = check_random_state(random_state)

    use_sparse = n_components < max_components - 1 and (n_samples * n_features) > 50_000
    U: np.ndarray | None = None
    s: np.ndarray | None = None

    if use_sparse:
        Z_scaled = Z.astype(np.float64) * scale
        seed = int(rs.randint(np.iinfo(np.int32).max))
        try:
            U_sparse, s_sparse, _ = sp.linalg.svds(
                Z_scaled,
                k=n_components,
                random_state=seed,
            )
            order = np.argsort(s_sparse)[::-1]
            U = U_sparse[:, order]
            s = s_sparse[order]
        except ArpackError, ArpackNoConvergence, ValueError:
            U = None
            s = None

    if U is None or s is None:
        Z_dense = np.asarray(Z.todense(), dtype=np.float64) * scale
        U_full, s_full, _ = np.linalg.svd(Z_dense, full_matrices=False)
        U = U_full[:, :n_components]
        s = s_full[:n_components]

    eigenvalues = np.asarray(s, dtype=np.float64) ** 2
    eigenvectors = np.asarray(U, dtype=np.float64)
    return LeafSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        n_estimators=int(n_estimators),
    )


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Compute the spectral-entropy effective rank of an eigenvalue list.

    Parameters
    ----------
    eigenvalues : array-like of shape (n_components,)
        Non-negative eigenvalues.

    Returns
    -------
    r_eff : float
        ``exp(H)`` where ``H = -Σ p_i log p_i`` and
        ``p_i = λ_i / Σ_j λ_j``. Returns ``0.0`` if all eigenvalues are
        non-positive.

    Notes
    -----
    Effective rank is a smoother analogue of matrix rank that is robust
    to small numerical perturbations of zero eigenvalues. See section 8
    of ``docs/random_forest_leaf_kernel_recipes.md`` for the framing
    used here.
    """
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    eigs = np.clip(eigs, 0.0, None)
    total = eigs.sum()
    if total <= 0.0:
        return 0.0
    p = eigs / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def inverse_participation_ratios(eigenvectors: np.ndarray) -> np.ndarray:
    """Compute inverse participation ratios per eigenvector.

    Parameters
    ----------
    eigenvectors : ndarray of shape (n_samples, n_components)
        Each column is treated as a single eigenvector. Columns are
        renormalized to unit Euclidean length before scoring so the IPR
        is invariant to scale.

    Returns
    -------
    ipr : ndarray of shape (n_components,)
        ``Σ_i v_i^4`` per (normalized) eigenvector. Small values
        (close to ``1 / n_samples``) indicate a broadly distributed
        mode; large values (close to ``1``) indicate a mode concentrated
        on a single sample.
    """
    V = np.asarray(eigenvectors, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError(
            "eigenvectors must be a 2-D array of shape "
            f"(n_samples, n_components); got shape {V.shape}."
        )
    norms = np.linalg.norm(V, axis=0, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    Vn = V / safe_norms
    return np.sum(Vn**4, axis=0)


def _relative_eigengaps(eigenvalues: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute relative eigengaps over a descending eigenvalue array."""
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    if eigs.size < 2:
        return np.zeros(0, dtype=np.float64)
    num = eigs[:-1] - eigs[1:]
    denom = eigs[1:] + eps
    return num / denom


class SpectralClusterCountSelector:
    """Eigengap-based ``n_clusters`` selector with modal-:math:`k` voting.

    The selector implements the cluster-count selection policy described
    in section 11 of ``docs/random_forest_leaf_kernel_recipes.md`` and
    in this library's design plan: relative eigengaps propose ``k``,
    several reseeded forests vote on the modal proposal, and the result
    falls back to ``k = 2`` rather than raising on degenerate spectra.

    Parameters
    ----------
    max_clusters : int, default=20
        Largest cluster count considered. The selector evaluates ``k``
        in ``range(2, max_clusters + 1)``.
    drop_leading_mode : bool, default=True
        If ``True``, the first (largest) eigenvalue is excluded from
        scoring. The leading eigenvector of a proximity kernel is
        usually a global "everything is connected" mode and dominates
        eigengaps without reflecting cluster structure.
    min_relative_gap : float, default=0.5
        Minimum relative eigengap required for a spectrum to be
        considered "structured." Spectra below this threshold do not
        contribute to the structure majority used for confidence
        scoring; if **no** spectrum is structured the selector falls
        back to ``k = 2`` and reports ``confidence = "low"``. The
        default is calibrated so that random-matrix-style noise
        (relative gaps under ~0.3) is rejected while compact-cluster
        kernels (relative gaps ≥ 1.0) are clearly retained.
    strategy : str, default="eigengap"
        Recorded on :class:`ClusterSelectionResult` for inspection.

    Notes
    -----
    The selector never raises on noisy or degenerate spectra; it always
    returns ``k >= 2``. This is what makes it compatible with sklearn
    ``check_estimator`` runs that exercise tiny synthetic datasets where
    eigengaps are essentially noise.

    The chosen ``n_clusters`` is the modal per-seed proposal **only**
    when the modal proposal is supported by a strict majority of input
    spectra and a strict majority of those spectra cleared
    ``min_relative_gap``. Otherwise the selector falls back to ``k = 2``
    and reports ``confidence = "low"``. The ``proposed_k_per_seed``
    field on :class:`ClusterSelectionResult` preserves the raw per-seed
    proposals for inspection regardless of whether the fallback was
    used.
    """

    def __init__(
        self,
        *,
        max_clusters: int = 20,
        drop_leading_mode: bool = True,
        min_relative_gap: float = 0.5,
        strategy: str = "eigengap",
    ) -> None:
        self.max_clusters = max_clusters
        self.drop_leading_mode = drop_leading_mode
        self.min_relative_gap = min_relative_gap
        self.strategy = strategy

    def select(self, spectra: list[LeafSpectrum]) -> ClusterSelectionResult:
        """Select ``n_clusters`` from one or more leaf-kernel spectra.

        Parameters
        ----------
        spectra : list of LeafSpectrum
            One spectrum per re-fitted forest seed. The first element is
            treated as the "primary" spectrum for the diagnostic fields
            of the returned result.

        Returns
        -------
        result : ClusterSelectionResult
            Selected ``n_clusters`` plus the supporting diagnostics.

        Raises
        ------
        ValueError
            If ``spectra`` is empty.
        """
        if not spectra:
            raise ValueError("spectra must contain at least one LeafSpectrum.")

        proposals = [self._propose_from_one(s.eigenvalues) for s in spectra]
        ks = np.array([p[0] for p in proposals], dtype=np.int64)
        gaps = np.array([p[1] for p in proposals], dtype=np.float64)
        structured = gaps >= self.min_relative_gap

        n_proposals = len(proposals)
        confidence: Literal["high", "low"] = "low"
        chosen_k = 2
        if bool(structured.any()):
            unique_vals, counts = np.unique(ks, return_counts=True)
            order = np.lexsort((unique_vals, -counts))
            modal_k = int(unique_vals[order[0]])
            modal_count = int(counts[order[0]])
            n_structured = int(structured.sum())
            has_modal_majority = 2 * modal_count > n_proposals
            has_structure_majority = 2 * n_structured > n_proposals
            if has_modal_majority and has_structure_majority:
                chosen_k = modal_k
                confidence = "high"

        primary = spectra[0]
        eigengaps = _relative_eigengaps(primary.eigenvalues)
        eff_rank = effective_rank(primary.eigenvalues)
        ipr = inverse_participation_ratios(primary.eigenvectors)

        return ClusterSelectionResult(
            n_clusters=chosen_k,
            eigenvalues=primary.eigenvalues.copy(),
            eigengaps=eigengaps,
            effective_rank=eff_rank,
            localization=ipr,
            proposed_k_per_seed=ks,
            strategy=self.strategy,
            confidence=confidence,
        )

    def _propose_from_one(self, eigenvalues: np.ndarray) -> tuple[int, float]:
        """Propose ``k`` and the supporting relative eigengap from one spectrum.

        With ``drop_leading_mode=True`` the largest eigenvalue is treated
        as a global "everything connected" mode and skipped. The
        remaining eigenvalues are scored at offsets ``k_eff = 1, 2, ...``
        (gap between ``λ_{k_eff}`` and ``λ_{k_eff+1}`` of the trimmed
        spectrum). The reported cluster count adds the dropped mode
        back, so ``k_eff = 1`` corresponds to ``n_clusters = 2``.
        """
        eigs = np.asarray(eigenvalues, dtype=np.float64)
        offset = 1 if (self.drop_leading_mode and eigs.size > 1) else 0
        eigs_eff = eigs[offset:]
        if eigs_eff.size < 2:
            return (2, 0.0)

        min_k_eff = max(1, 2 - offset)
        max_k_eff = int(min(self.max_clusters - offset, eigs_eff.size - 1))
        if max_k_eff < min_k_eff:
            return (2, 0.0)

        candidate_k_eff = np.arange(min_k_eff, max_k_eff + 1)
        num = eigs_eff[candidate_k_eff - 1] - eigs_eff[candidate_k_eff]
        denom = eigs_eff[candidate_k_eff] + 1e-12
        gaps = num / denom

        idx = int(np.argmax(gaps))
        proposed_k = int(candidate_k_eff[idx]) + offset
        return (proposed_k, float(gaps[idx]))
