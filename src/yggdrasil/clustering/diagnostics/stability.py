"""Label-stability diagnostics across resampled fits.

The :class:`yggdrasil.clustering.SpectralClusterCountSelector` evaluates
each candidate ``k`` on multiple reseeded forests. These helpers
reduce the resulting list of labelings into a single stability score
(the mean off-diagonal pairwise ARI or NMI) and also expose the full
pairwise matrix for inspection.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

__all__ = [
    "label_stability",
    "pairwise_ari_matrix",
]


def _coerce_labelings(labelings: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Return labelings as a list of 1-D ndarrays of identical length."""
    coerced = [np.asarray(lbl).ravel() for lbl in labelings]
    if not coerced:
        return coerced
    n = coerced[0].size
    for i, lbl in enumerate(coerced):
        if lbl.size != n:
            raise ValueError(
                "All labelings must have the same length; "
                f"labelings[0] has size {n} but labelings[{i}] has size {lbl.size}."
            )
    return coerced


def pairwise_ari_matrix(
    labelings: Sequence[np.ndarray],
    metric: Literal["ari", "nmi"] = "ari",
) -> np.ndarray:
    """Compute the pairwise agreement matrix of a list of labelings.

    Parameters
    ----------
    labelings : sequence of array-like of shape (n_samples,)
        One labeling per resampled fit; all labelings must have the
        same length.
    metric : {"ari", "nmi"}, default="ari"
        Pairwise agreement score: adjusted Rand index (default) or
        normalized mutual information.

    Returns
    -------
    M : ndarray of shape (n_resamples, n_resamples)
        Symmetric matrix with ``M[i, i] = 1.0`` and ``M[i, j]`` equal
        to the chosen agreement score between labelings ``i`` and
        ``j``. Returns the empty array when ``labelings`` is empty.

    Raises
    ------
    ValueError
        If ``metric`` is not one of ``{"ari", "nmi"}`` or if labelings
        have different lengths.
    """
    if metric not in {"ari", "nmi"}:
        raise ValueError(f"metric must be 'ari' or 'nmi'; got {metric!r}.")

    coerced = _coerce_labelings(labelings)
    n = len(coerced)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    score = adjusted_rand_score if metric == "ari" else normalized_mutual_info_score
    M = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(score(coerced[i], coerced[j]))
            M[i, j] = value
            M[j, i] = value
    return M


def label_stability(
    labelings: Sequence[np.ndarray],
    metric: Literal["ari", "nmi"] = "ari",
) -> float:
    """Mean pairwise agreement across a list of labelings.

    Parameters
    ----------
    labelings : sequence of array-like of shape (n_samples,)
        One labeling per resampled fit; all labelings must have the
        same length.
    metric : {"ari", "nmi"}, default="ari"
        Pairwise agreement score; see :func:`pairwise_ari_matrix`.

    Returns
    -------
    stability : float
        Mean of the strict-upper-triangle entries of the pairwise
        agreement matrix. Returns ``nan`` when fewer than two
        labelings are supplied (no off-diagonal entries to average).

    Examples
    --------
    >>> import numpy as np
    >>> from yggdrasil.clustering.diagnostics import label_stability
    >>> a = np.array([0, 0, 1, 1])
    >>> b = np.array([1, 1, 0, 0])
    >>> float(label_stability([a, b]))
    1.0
    """
    coerced = _coerce_labelings(labelings)
    n = len(coerced)
    if n < 2:
        return float("nan")
    M = pairwise_ari_matrix(coerced, metric=metric)
    iu = np.triu_indices(n, k=1)
    return float(np.mean(M[iu]))
