"""Spectral diagnostics on the leaf kernel ``K = Z Zᵀ / T``.

This module provides the spectrum-only pieces that summarize the
leaf-kernel structure without making cluster-count decisions:

- :class:`LeafSpectrum` packages the top eigenvalues, eigenvectors, and
  the kernel normalization constant returned by
  :func:`compute_leaf_spectrum`.
- :func:`effective_rank` measures spectral concentration via the
  Shannon entropy of the normalized eigenvalues.
- :func:`inverse_participation_ratios` measures how localized each
  eigenvector is across samples.
- :func:`cumulative_spectral_mass` returns the smallest ``k`` whose
  leading eigenvalues capture a target fraction of the total spectral
  mass.
- :func:`eigengap_curve` returns the relative-eigengap series used by
  the :mod:`yggdrasil.clustering.selector` module.

Selection logic that consumes these summaries lives in
:mod:`yggdrasil.clustering.selector`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence
from sklearn.utils import check_random_state

__all__ = [
    "LeafSpectrum",
    "compute_leaf_spectrum",
    "cumulative_spectral_mass",
    "effective_rank",
    "eigengap_curve",
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


def cumulative_spectral_mass(
    spectrum: LeafSpectrum | np.ndarray,
    threshold: float = 0.9,
) -> int:
    """Return the smallest ``k`` whose top ``k`` eigenvalues exceed a mass fraction.

    Computes ``cumsum(λ) / sum(λ)`` over a descending eigenvalue array
    and returns the smallest index (1-based) at which the cumulative
    fraction is ``>= threshold``. Useful as a complement to
    :func:`effective_rank`: where effective rank reports a smooth
    "number of meaningful directions", cumulative mass reports the
    hard cutoff at a chosen energy level.

    Parameters
    ----------
    spectrum : LeafSpectrum or array-like of shape (n_components,)
        Either a :class:`LeafSpectrum` (in which case its
        ``eigenvalues`` array is used) or a raw descending eigenvalue
        array.
    threshold : float, default=0.9
        Target fraction of total spectral mass; must satisfy
        ``0 < threshold <= 1``.

    Returns
    -------
    k : int
        Smallest ``k`` with ``sum(λ[:k]) / sum(λ) >= threshold``.
        Always at least ``1``.

    Raises
    ------
    ValueError
        If ``threshold`` is outside ``(0, 1]`` or if the eigenvalue
        array is empty or has non-positive total mass.

    Examples
    --------
    >>> import numpy as np
    >>> from yggdrasil.clustering.diagnostics import cumulative_spectral_mass
    >>> cumulative_spectral_mass(np.array([4.0, 1.0, 1.0]), threshold=0.6)
    1
    >>> cumulative_spectral_mass(np.array([4.0, 1.0, 1.0]), threshold=0.9)
    3
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must lie in (0, 1]; got {threshold}.")

    if isinstance(spectrum, LeafSpectrum):
        eigs = np.asarray(spectrum.eigenvalues, dtype=np.float64)
    else:
        eigs = np.asarray(spectrum, dtype=np.float64)

    if eigs.size == 0:
        raise ValueError("eigenvalue array must be non-empty.")

    eigs = np.clip(eigs, 0.0, None)
    total = float(eigs.sum())
    if total <= 0.0:
        raise ValueError("eigenvalue array has non-positive total mass.")

    fractions = np.cumsum(eigs) / total
    idx = int(np.searchsorted(fractions, threshold, side="left"))
    idx = min(idx, eigs.size - 1)
    return idx + 1


def eigengap_curve(
    spectrum: LeafSpectrum | np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute the relative-eigengap series of a descending eigenvalue array.

    The relative gap at index ``i`` is ``(λ_i - λ_{i+1}) / (λ_{i+1} + ε)``
    so that gaps remain comparable across spectra of different scales.
    The selector module uses this series as one of its candidate-set
    sources.

    Parameters
    ----------
    spectrum : LeafSpectrum or array-like of shape (n_components,)
        Either a :class:`LeafSpectrum` or a raw descending eigenvalue
        array.
    eps : float, default=1e-12
        Numerical floor added to the denominator to keep gaps finite
        when the trailing eigenvalues are near zero.

    Returns
    -------
    gaps : ndarray of shape (n_components - 1,)
        Relative eigengaps. Returns an empty array when fewer than two
        eigenvalues are supplied.

    Examples
    --------
    >>> import numpy as np
    >>> from yggdrasil.clustering.diagnostics import eigengap_curve
    >>> eigengap_curve(np.array([4.0, 1.0, 0.5])).round(2)
    array([3., 1.])
    """
    if isinstance(spectrum, LeafSpectrum):
        eigs = np.asarray(spectrum.eigenvalues, dtype=np.float64)
    else:
        eigs = np.asarray(spectrum, dtype=np.float64)

    if eigs.size < 2:
        return np.zeros(0, dtype=np.float64)
    num = eigs[:-1] - eigs[1:]
    denom = eigs[1:] + eps
    return num / denom
