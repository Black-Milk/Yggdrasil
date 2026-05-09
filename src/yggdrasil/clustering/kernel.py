"""Leaf-kernel utilities for forest embeddings.

These helpers convert the leaf-index assignments produced by a fitted
tree ensemble into a sparse leaf-indicator matrix ``Z``, the
corresponding proximity kernel ``K = Z Zᵀ / T``, and the matching
dissimilarity ``D = 1 - K``.

Conventions follow scikit-learn's :meth:`forest.apply` output:
``leaves[i, t]`` is the leaf id reached by sample ``i`` in tree ``t``,
and ``T`` denotes the number of trees in the ensemble.

The proximity kernel is the standard Shi-Horvath random-forest similarity
measure; two samples are similar when they fall into the same terminal
leaf in many trees.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

__all__ = [
    "kernel_from_leaves",
    "leaf_dissimilarity",
    "leaf_indicator_matrix",
    "leaf_kernel",
]


def leaf_indicator_matrix(leaves: np.ndarray) -> sp.csr_matrix:
    """Build a sparse leaf-indicator matrix from forest leaf assignments.

    Each row encodes the leaves a single sample reaches, with one nonzero
    per tree. Leaf ids are made globally unique by treating each tree's
    column as a separate categorical feature, so column ``j`` corresponds
    to a single ``(tree, leaf)`` pair across the whole forest.

    Parameters
    ----------
    leaves : array-like of shape (n_samples, n_estimators)
        Leaf ids returned by ``forest.apply(X)`` for a fitted ensemble.

    Returns
    -------
    Z : sparse CSR matrix of shape (n_samples, n_total_leaves)
        Binary indicator matrix with exactly ``n_estimators`` ones per
        row. ``n_total_leaves`` is the sum of distinct leaf-id counts
        across trees.

    Raises
    ------
    ValueError
        If ``leaves`` is not a 2-D array.

    Notes
    -----
    Raw leaf ids returned by ``forest.apply`` are tree-local: a leaf id
    of ``5`` in one tree has no relation to ``5`` in another. This helper
    offsets ids so each ``(tree, leaf)`` pair gets a unique global column.

    Examples
    --------
    >>> import numpy as np
    >>> from yggdrasil.clustering.kernel import leaf_indicator_matrix
    >>> leaves = np.array([[0, 1], [0, 2], [1, 1]])
    >>> Z = leaf_indicator_matrix(leaves)
    >>> Z.shape
    (3, 5)
    >>> int(Z.sum())
    6
    """
    leaves = np.asarray(leaves)
    if leaves.ndim != 2:
        raise ValueError(
            "leaves must be a 2-D array of shape (n_samples, n_estimators); "
            f"got shape {leaves.shape}."
        )
    encoder = OneHotEncoder(sparse_output=True, dtype=np.float64)
    return encoder.fit_transform(leaves).tocsr()


def leaf_kernel(
    Z: sp.spmatrix | np.ndarray,
    n_estimators: int | None = None,
) -> np.ndarray:
    """Compute the forest leaf-proximity kernel ``K = Z Zᵀ / T``.

    Parameters
    ----------
    Z : {array-like, sparse matrix} of shape (n_samples, n_total_leaves)
        Leaf-indicator matrix; each row should have exactly
        ``n_estimators`` ones (one per tree).
    n_estimators : int, optional
        Number of trees in the forest. If ``None``, inferred from the row
        sums of ``Z``, which must be constant across rows.

    Returns
    -------
    K : ndarray of shape (n_samples, n_samples)
        Proximity matrix in ``[0, 1]``; ``K[i, j]`` is the fraction of
        trees in which samples ``i`` and ``j`` share a terminal leaf.

    Raises
    ------
    ValueError
        If ``n_estimators`` is not provided and the row sums of ``Z`` are
        not constant, or if ``n_estimators`` is non-positive.

    Notes
    -----
    The materialized kernel is dense and consumes
    :math:`O(n_{\\mathrm{samples}}^2)` memory. For large inputs prefer
    operating on the sparse ``Z`` directly via SVD; see
    :func:`yggdrasil.clustering.diagnostics.compute_leaf_spectrum`.
    """
    if not sp.issparse(Z):
        Z = sp.csr_matrix(Z)

    if n_estimators is None:
        row_sums = np.asarray(Z.sum(axis=1)).ravel()
        unique = np.unique(row_sums)
        if unique.size != 1:
            raise ValueError(
                "n_estimators is None but Z row sums are not constant; "
                "cannot infer the number of trees. Pass n_estimators "
                "explicitly."
            )
        n_estimators = int(unique[0])

    if n_estimators <= 0:
        raise ValueError(f"n_estimators must be a positive integer; got {n_estimators}.")

    K = (Z @ Z.T).toarray().astype(np.float64) / n_estimators
    # Z @ Z.T is symmetric in exact arithmetic, but the sparse multiply
    # accumulates entries (i, j) and (j, i) in different orders and can
    # leave a few-ULP asymmetry. Project onto the symmetric subspace so
    # downstream consumers (eigh, check_symmetric, spectral clustering)
    # see bit-exact symmetry.
    K = 0.5 * (K + K.T)
    return K


def leaf_dissimilarity(K: np.ndarray) -> np.ndarray:
    """Convert a leaf-proximity kernel to a dissimilarity matrix.

    Parameters
    ----------
    K : ndarray of shape (n_samples, n_samples)
        Proximity matrix with values in ``[0, 1]``.

    Returns
    -------
    D : ndarray of shape (n_samples, n_samples)
        Dissimilarity ``D = 1 - K`` with zero diagonal up to
        floating-point precision.
    """
    return 1.0 - np.asarray(K, dtype=np.float64)


def kernel_from_leaves(leaves: np.ndarray) -> np.ndarray:
    """Compute the leaf-proximity kernel directly from raw leaf assignments.

    Convenience wrapper that composes :func:`leaf_indicator_matrix` and
    :func:`leaf_kernel` so callers with raw ``forest.apply(X)`` output do
    not have to manage the one-hot encoding themselves.

    Parameters
    ----------
    leaves : array-like of shape (n_samples, n_estimators)
        Leaf ids returned by ``forest.apply(X)``.

    Returns
    -------
    K : ndarray of shape (n_samples, n_samples)
        Proximity matrix in ``[0, 1]``.
    """
    leaves = np.asarray(leaves)
    Z = leaf_indicator_matrix(leaves)
    return leaf_kernel(Z, n_estimators=int(leaves.shape[1]))
