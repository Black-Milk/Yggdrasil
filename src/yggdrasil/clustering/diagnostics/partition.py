"""Partition-quality metrics for spectral embeddings and leaf kernels.

These functions act on a candidate labeling rather than on the
spectrum: silhouette in the spectral embedding space, and Newman
modularity on the leaf-proximity kernel viewed as a weighted
adjacency matrix. Both are consumed by
:class:`yggdrasil.clustering.SpectralClusterCountSelector` in composite
mode but are also useful as standalone diagnostics on any embedding or
kernel.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score

__all__ = [
    "modularity_on_kernel",
    "silhouette_on_embedding",
]


def silhouette_on_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> float:
    """Compute the mean silhouette coefficient on a spectral embedding.

    Thin wrapper around :func:`sklearn.metrics.silhouette_score` with
    Euclidean distance, intended to be called on the row-normalized
    top eigenvectors of a leaf kernel. Returns a float in ``[-1, 1]``;
    higher is better. Returns ``nan`` when fewer than two distinct
    labels are present (the metric is undefined in that case rather
    than an error, so the selector can carry the candidate without
    raising).

    Parameters
    ----------
    embedding : ndarray of shape (n_samples, n_components)
        Spectral or otherwise low-dimensional embedding of the data.
    labels : array-like of shape (n_samples,)
        Cluster index per sample.
    sample_size : int, optional
        If not ``None``, compute the silhouette on a random subset of
        rows. Forwarded to
        :func:`sklearn.metrics.silhouette_score`.
    random_state : int, RandomState instance or None, default=None
        Random state used by ``sample_size`` subsampling. See
        :term:`Glossary <random_state>`.

    Returns
    -------
    score : float
        Mean silhouette coefficient, or ``nan`` when fewer than two
        distinct labels are present.

    Examples
    --------
    >>> import numpy as np
    >>> from yggdrasil.clustering.diagnostics import silhouette_on_embedding
    >>> rng = np.random.default_rng(0)
    >>> emb = np.vstack([rng.normal(0.0, 0.05, (10, 2)), rng.normal(5.0, 0.05, (10, 2))])
    >>> labels = np.array([0] * 10 + [1] * 10)
    >>> float(silhouette_on_embedding(emb, labels)) > 0.9
    True
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(
        silhouette_score(
            embedding,
            labels,
            metric="euclidean",
            sample_size=sample_size,
            random_state=random_state,
        )
    )


def modularity_on_kernel(K: np.ndarray, labels: np.ndarray) -> float:
    """Compute Newman modularity of a partition on a kernel-as-adjacency.

    Treats ``K`` as a weighted, undirected adjacency matrix, zeroes the
    diagonal (self-loops are not informative for community structure),
    and returns

    .. math::

        Q = \\frac{1}{2m} \\sum_{ij} \\Bigl(A_{ij} - \\frac{k_i k_j}{2m}\\Bigr)
            \\mathbb{1}[c_i = c_j]

    where ``A`` is the de-diagonalized kernel, ``k_i`` is the row sum,
    and ``2m`` is the total edge weight. Higher is better; ``Q`` lies
    in ``[-0.5, 1]``.

    Parameters
    ----------
    K : ndarray of shape (n_samples, n_samples)
        Symmetric, non-negative similarity matrix. The leaf kernel
        produced by :func:`yggdrasil.clustering.kernel.leaf_kernel`
        satisfies these conditions.
    labels : array-like of shape (n_samples,)
        Cluster index per sample.

    Returns
    -------
    Q : float
        Newman modularity. Returns ``0.0`` when the de-diagonalized
        kernel has no positive edge weight.

    Notes
    -----
    Negative weights would produce a non-standard modularity; this
    function clips negative entries to zero so the formula stays
    bounded in ``[-0.5, 1]`` for any input.

    Examples
    --------
    >>> import numpy as np
    >>> from yggdrasil.clustering.diagnostics import modularity_on_kernel
    >>> K = np.array(
    ...     [
    ...         [1.0, 0.9, 0.0, 0.0],
    ...         [0.9, 1.0, 0.0, 0.0],
    ...         [0.0, 0.0, 1.0, 0.9],
    ...         [0.0, 0.0, 0.9, 1.0],
    ...     ]
    ... )
    >>> good = modularity_on_kernel(K, np.array([0, 0, 1, 1]))
    >>> bad = modularity_on_kernel(K, np.array([0, 1, 0, 1]))
    >>> bool(good > bad)
    True
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square (n_samples, n_samples); got shape {K.shape}.")
    labels = np.asarray(labels)
    if labels.shape[0] != K.shape[0]:
        raise ValueError(
            f"labels length must equal K.shape[0]; got {labels.shape[0]} and {K.shape[0]}."
        )

    A = np.clip(K, 0.0, None).copy()
    np.fill_diagonal(A, 0.0)
    two_m = float(A.sum())
    if two_m <= 0.0:
        return 0.0

    degrees = A.sum(axis=1)
    modularity = 0.0
    for c in np.unique(labels):
        mask = labels == c
        in_weight = float(A[np.ix_(mask, mask)].sum())
        deg_sum = float(degrees[mask].sum())
        modularity += in_weight / two_m - (deg_sum / two_m) ** 2
    return float(modularity)
