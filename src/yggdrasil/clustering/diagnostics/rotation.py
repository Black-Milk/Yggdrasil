"""Zelnik-Manor and Perona rotation cost on top eigenvectors.

The "self-tuning spectral clustering" diagnostic optimizes a rotation
:math:`R \\in SO(k)` of the row-normalized top-k eigenvectors so that
each row becomes as close to a 1-hot indicator as possible. The
optimal cost is small when the eigenvectors already encode a clean
partition into ``k`` clusters and large when they do not, which makes
it a far more discriminative spectrum-shape signal than the raw
eigengap.

References
----------
Zelnik-Manor, L., & Perona, P. (2004). "Self-tuning spectral
clustering." *Advances in Neural Information Processing Systems*, 17.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.utils import check_random_state

__all__ = ["rotation_cost"]


def _row_normalize(X: np.ndarray) -> np.ndarray:
    """Normalize each row to unit Euclidean length, leaving zero rows alone."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return X / safe


def _givens_rotation_product(theta: np.ndarray, k: int) -> np.ndarray:
    """Build a rotation matrix as a product of Givens rotations.

    Parameters
    ----------
    theta : ndarray of shape (k * (k - 1) / 2,)
        Rotation angles, one per ``(i, j)`` plane with ``i < j``.
    k : int
        Embedding dimensionality.

    Returns
    -------
    R : ndarray of shape (k, k)
        Rotation matrix in ``SO(k)``.
    """
    R = np.eye(k, dtype=np.float64)
    idx = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            angle = float(theta[idx])
            c = np.cos(angle)
            s = np.sin(angle)
            G = np.eye(k, dtype=np.float64)
            G[i, i] = c
            G[j, j] = c
            G[i, j] = -s
            G[j, i] = s
            R = R @ G
            idx += 1
    return R


def _alignment_cost(theta: np.ndarray, X: np.ndarray) -> float:
    """Raw Zelnik-Manor and Perona alignment cost at rotation ``theta``.

    Returns ``(1/n) Σ_i Σ_j (Y_{ij} / M_i)^2`` where ``Y = X R`` and
    ``M_i = max_j |Y_{ij}|``. With row-normalized ``X`` this lives in
    ``[1, k]``: every row's dominant coordinate contributes ``1``, and
    the remaining ``k - 1`` coordinates contribute their squared
    "leakage" relative to the dominant coordinate. The k-rescaling to
    ``[0, 1]`` happens in :func:`rotation_cost`.
    """
    n, k = X.shape
    R = _givens_rotation_product(theta, k)
    Y = X @ R
    abs_Y = np.abs(Y)
    max_per_row = abs_Y.max(axis=1)
    safe = np.where(max_per_row > 0, max_per_row, 1.0)
    return float(np.sum((Y / safe[:, None]) ** 2)) / max(n, 1)


def rotation_cost(
    top_eigvecs: np.ndarray,
    *,
    n_restarts: int = 5,
    max_iter: int = 200,
    tol: float = 1e-6,
    random_state: int | np.random.RandomState | None = None,
) -> float:
    """Zelnik-Manor and Perona rotation cost of an eigenvector matrix.

    Row-normalizes the input, parameterizes a rotation ``R \\in SO(k)``
    by ``k(k-1)/2`` Givens angles, and minimizes the alignment cost
    via L-BFGS-B with multiple random restarts. The raw alignment cost
    ``J(R)/n`` lives in ``[1, k]`` (the lower bound is hit when every
    row is a perfect 1-hot, the upper bound when rows are uniform);
    this function returns the rescaled value
    ``(J(R)/n - 1) / (k - 1)`` so different ``k`` are on the same
    ``[0, 1]`` axis. Lower is better.

    Parameters
    ----------
    top_eigvecs : ndarray of shape (n_samples, k)
        Leading eigenvectors of the leaf kernel, with ``k`` equal to
        the candidate cluster count under evaluation.
    n_restarts : int, default=5
        Number of L-BFGS-B restarts from independent random
        initializations. The best (lowest) alignment cost is returned.
        The first restart starts near the identity rotation; the
        remaining ones sample angles broadly to break out of local
        minima. Set to ``1`` to recover the cheapest single-shot
        behavior.
    max_iter : int, default=200
        Maximum L-BFGS-B iterations per restart.
    tol : float, default=1e-6
        Gradient-norm tolerance for L-BFGS-B convergence.
    random_state : int, RandomState instance or None, default=None
        Controls the random initializations. See
        :term:`Glossary <random_state>`.

    Returns
    -------
    cost : float
        Rescaled alignment cost in ``[0, 1]``: ``0`` when the
        eigenvectors are already a clean indicator basis, ``1`` when
        every row is uniformly spread across the ``k`` dimensions.
        Returns ``0.0`` for ``k < 2`` (trivial case).

    Notes
    -----
    The optimization is local; the alignment cost is non-convex in the
    Givens angles. The signed-permutation symmetry of ``SO(k)`` means
    every clean clustering corresponds to ``2^k k!`` equivalent global
    minima, plus genuinely worse local minima from spurious
    near-uniform rows. Restarts with broad initial angles are the
    standard countermeasure (Zelnik-Manor and Perona, 2004). For
    well-clustered data a single restart usually suffices; the default
    of ``5`` is cheap insurance for noisy spectra and large ``k``.

    Examples
    --------
    >>> import numpy as np
    >>> from yggdrasil.clustering.diagnostics import rotation_cost
    >>> indicator = np.eye(3, dtype=np.float64)[np.repeat([0, 1, 2], 4)]
    >>> rotation_cost(indicator, random_state=0) < 0.05
    True
    """
    X = np.asarray(top_eigvecs, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"top_eigvecs must be a 2-D array of shape (n_samples, k); got shape {X.shape}."
        )
    if int(n_restarts) < 1:
        raise ValueError(f"n_restarts must be at least 1; got {n_restarts}.")
    _, k = X.shape
    if k < 2:
        return 0.0

    X_norm = _row_normalize(X)
    n_angles = k * (k - 1) // 2
    rs = check_random_state(random_state)
    options = {"maxiter": int(max_iter), "gtol": float(tol)}

    best = float("inf")
    for restart in range(int(n_restarts)):
        if restart == 0:
            theta0 = rs.normal(scale=0.01, size=n_angles)
        else:
            theta0 = rs.uniform(low=-np.pi, high=np.pi, size=n_angles)
        result = minimize(
            _alignment_cost,
            theta0,
            args=(X_norm,),
            method="L-BFGS-B",
            options=options,
        )
        if float(result.fun) < best:
            best = float(result.fun)

    rescaled = (best - 1.0) / max(k - 1, 1)
    return float(np.clip(rescaled, 0.0, 1.0))
