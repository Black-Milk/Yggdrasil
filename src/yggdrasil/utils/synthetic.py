"""Synthetic-data sampling utilities for unsupervised tree-ensemble methods.

These primitives sample from the *empirical marginal distribution* of a feature
matrix. They form the backbone of the "real vs. synthetic" construction used
by unsupervised random forests (Shi & Horvath, 2006), where a discriminator
is trained to separate the observed data from columnwise-shuffled / uniformly-
sampled surrogates.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.utils import check_random_state

SamplingMethod = Literal["bootstrap", "permutation", "uniform"]

__all__ = [
    "SamplingMethod",
    "bootstrap_sample_column",
    "generate_discriminative_dataset",
    "generate_synthetic_features",
    "permutation_sample_column",
    "uniform_sample_column",
]


def bootstrap_sample_column(
    X: np.ndarray,
    n_samples: int | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Bootstrap-sample a single column.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples,)
        Column to bootstrap.
    n_samples : int, optional
        Number of samples to draw. Defaults to ``X.shape[0]``.
    random_state : int, RandomState instance, or None
        Seed or generator for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        The bootstrapped column.
    """
    rng = check_random_state(random_state)
    if n_samples is None:
        n_samples = X.shape[0]
    return rng.choice(X, size=n_samples, replace=True)


def uniform_sample_column(
    X: np.ndarray,
    n_samples: int | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Sample uniformly between the min and max of a column.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples,)
        Column whose support defines the sampling range.
    n_samples : int, optional
        Number of samples to draw. Defaults to ``X.shape[0]``.
    random_state : int, RandomState instance, or None
        Seed or generator for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Uniformly sampled column.
    """
    rng = check_random_state(random_state)
    if n_samples is None:
        n_samples = X.shape[0]
    return rng.uniform(np.min(X), np.max(X), size=n_samples)


def permutation_sample_column(
    X: np.ndarray,
    n_samples: int | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Sample a single column without replacement.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples,)
        Column to permute.
    n_samples : int, optional
        Number of samples to draw. Defaults to ``X.shape[0]``.
    random_state : int, RandomState instance, or None
        Seed or generator for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        The permuted column.

    Raises
    ------
    ValueError
        If ``n_samples`` is larger than ``X.shape[0]``.
    """
    rng = check_random_state(random_state)
    if n_samples is None:
        n_samples = X.shape[0]
    if n_samples > X.shape[0]:
        raise ValueError("n_samples cannot exceed X.shape[0] for permutation sampling.")
    return rng.permutation(X)[:n_samples]


def generate_synthetic_features(
    X: np.ndarray,
    method: SamplingMethod = "bootstrap",
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Generate a synthetic feature matrix from the empirical marginals of ``X``.

    Each output column is sampled independently from the corresponding column of
    ``X``, breaking the joint dependence structure between features. This is the
    standard surrogate construction used by unsupervised random forests.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Reference feature matrix.
    method : {"bootstrap", "permutation", "uniform"}, default="bootstrap"
        ``"bootstrap"`` samples each column with replacement; ``"permutation"``
        independently shuffles each column without replacement; ``"uniform"``
        draws each column uniformly between its min and max.
    random_state : int, RandomState instance, or None
        Seed or generator for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        The synthetic feature matrix.
    """
    rng = check_random_state(random_state)
    n_features = int(X.shape[1])
    synth_X = np.empty_like(X)
    for column in range(n_features):
        if method == "bootstrap":
            synth_X[:, column] = bootstrap_sample_column(X[:, column], random_state=rng)
        elif method == "permutation":
            synth_X[:, column] = permutation_sample_column(X[:, column], random_state=rng)
        elif method == "uniform":
            synth_X[:, column] = uniform_sample_column(X[:, column], random_state=rng)
        else:
            raise ValueError("method must be one of 'bootstrap', 'permutation', or 'uniform'.")
    return synth_X


def generate_discriminative_dataset(
    X: np.ndarray,
    method: SamplingMethod = "bootstrap",
    random_state: int | np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a "real vs. synthetic" labelled dataset.

    Real rows from ``X`` are labelled ``1`` and an equal number of synthetic
    rows (drawn via :func:`generate_synthetic_features`) are labelled ``0``.
    The combined rows are randomly permuted so the two populations are
    intermixed.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Reference feature matrix.
    method : {"bootstrap", "permutation", "uniform"}, default="bootstrap"
        Synthetic-sampling method; see :func:`generate_synthetic_features`.
    random_state : int, RandomState instance, or None
        Seed or generator for reproducibility.

    Returns
    -------
    X_out : np.ndarray of shape (2 * n_samples, n_features)
        Stacked real and synthetic rows, randomly permuted.
    y_out : np.ndarray of shape (2 * n_samples,)
        Labels: ``1`` for real, ``0`` for synthetic.
    """
    rng = check_random_state(random_state)
    n_samples = int(X.shape[0])

    synth_X = generate_synthetic_features(X, method=method, random_state=rng)
    X_out = np.vstack((X, synth_X))
    y_out = np.concatenate((np.ones(n_samples), np.zeros(n_samples)))

    perm = rng.permutation(X_out.shape[0])
    return X_out[perm], y_out[perm]
