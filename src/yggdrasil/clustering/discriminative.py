"""Auto-:math:`k` clustering on discriminative-forest leaf kernels.

:class:`DiscriminativeForestClusterer` wraps the existing real-vs-synthetic
:class:`~yggdrasil.clustering.DiscriminativeForestEmbedding` with an
inspectable cluster-count selection layer and a spectral-embedding
``k``-means label assignment. Internally it computes the leaf-kernel
spectrum once via truncated SVD on the sparse leaf-indicator matrix,
shares that spectrum between diagnostics and the embedding, optionally
votes on ``n_clusters`` across a small number of reseeded forests, and
falls back to ``k = 2`` rather than raising on noisy spectra.

The synthetic samples used by the discriminator forest are training
contrast only; they are never clustered. After fitting the forest, the
clusterer applies it back to the original real rows and operates on the
resulting leaf embedding.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_random_state, validate_data

from yggdrasil.clustering.diagnostics import (
    ClusterSelectionResult,
    LeafSpectrum,
    SpectralClusterCountSelector,
    _relative_eigengaps,
    compute_leaf_spectrum,
    effective_rank,
    inverse_participation_ratios,
)
from yggdrasil.clustering.forest import Backend, DiscriminativeForestEmbedding
from yggdrasil.utils.synthetic import SamplingMethod

__all__ = ["DiscriminativeForestClusterer"]


class DiscriminativeForestClusterer(ClusterMixin, BaseEstimator):
    """Clustering via discriminative forest leaf kernel with auto-:math:`k`.

    Trains an internal :class:`DiscriminativeForestEmbedding` to separate
    real samples from synthetic samples drawn from their empirical
    marginals, builds the leaf-kernel spectrum from the sparse
    leaf-indicator matrix of the real rows, selects ``n_clusters_`` from
    that spectrum (or accepts a user-provided integer), and assigns
    labels by running k-means on the top eigenvectors of the kernel.

    The cluster-count selector is intentionally defensive: it scores
    relative eigengaps after optionally dropping the leading global
    mode, takes the modal proposal across a few reseeded forests, and
    falls back to ``k = 2`` with low confidence rather than raising
    when the spectrum is essentially noise.

    Parameters
    ----------
    n_clusters : int or "auto", default="auto"
        Number of clusters. If ``"auto"``, ``n_clusters_`` is selected
        from the leaf-kernel spectrum.
    max_clusters : int, default=20
        Largest cluster count considered when ``n_clusters="auto"``.
        Ignored when ``n_clusters`` is an explicit integer.
    cluster_selection : {"eigengap"}, default="eigengap"
        Cluster-count selection strategy. Only ``"eigengap"`` is
        implemented in this version.
    drop_leading_mode : bool, default=True
        If ``True``, the largest eigenvalue/eigenvector pair is excluded
        when scoring eigengaps and when forming the spectral embedding.
        The leading mode of a forest proximity kernel is usually a
        global "everything connected" component that obscures cluster
        structure.
    n_selection_resamples : int, default=3
        Number of forests fit during ``n_clusters="auto"`` selection,
        each with a different seed; the modal proposed ``k`` is chosen.
        Ignored when ``n_clusters`` is an explicit integer.
    n_estimators : int, default=100
        Number of trees in each fitted forest.
    backend : {"random_forest", "extra_trees"}, default="random_forest"
        Forest implementation forwarded to
        :class:`DiscriminativeForestEmbedding`.
    synthetic_method : {"bootstrap", "permutation", "uniform"}, default="bootstrap"
        Synthetic-sampling method used by the discriminator forest.
    random_state : int, RandomState instance or None, default=None
        Controls all randomness: synthetic-sample generation, forest
        seeds, the SVD initialization, and the final k-means.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster index assigned to each training sample.
    n_clusters_ : int
        Number of clusters used for the final assignment. Equals
        ``n_clusters`` when an integer was provided; otherwise the
        eigengap-selected count.
    spectral_embedding_ : ndarray of shape (n_samples, n_components)
        Row-normalized matrix of top eigenvectors used as the input to
        k-means. ``n_components <= n_clusters_`` (one less when
        ``drop_leading_mode=True`` and the leading mode was excluded).
    forest_embedding_ : DiscriminativeForestEmbedding
        The fitted underlying embedding. Useful for inspecting the
        forest, the synthetic data, and the leaf assignments.
    cluster_selection_ : ClusterSelectionResult
        Structured summary of how ``n_clusters_`` was chosen, including
        eigenvalues, eigengaps, effective rank, and per-seed proposals.
    random_state_ : RandomState
        The random number generator instantiated from ``random_state``.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during :meth:`fit`. Defined only when
        ``X`` has feature names that are all strings.

    Notes
    -----
    The spectrum used for both selection and label assignment is
    computed once via truncated SVD on the sparse leaf-indicator matrix;
    the dense :math:`n \\times n` kernel is never materialized.

    Kernel ``k``-means against the precomputed leaf kernel is documented
    as an alternative in
    ``docs/random_forest_leaf_kernel_recipes.md`` but is not provided as
    a backend in this version.

    Examples
    --------
    >>> import numpy as np
    >>> from yggdrasil.clustering import DiscriminativeForestClusterer
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack(
    ...     [
    ...         rng.normal(loc=0.0, scale=0.3, size=(20, 4)),
    ...         rng.normal(loc=5.0, scale=0.3, size=(20, 4)),
    ...     ]
    ... )
    >>> est = DiscriminativeForestClusterer(
    ...     n_estimators=20, n_selection_resamples=1, random_state=0
    ... ).fit(X)
    >>> est.labels_.shape
    (40,)
    >>> 1 <= est.n_clusters_ <= 20
    True
    """

    _parameter_constraints: dict[str, list[Any]] = {
        "n_clusters": [Integral, StrOptions({"auto"})],
        "max_clusters": [Interval(Integral, 2, None, closed="left")],
        "cluster_selection": [StrOptions({"eigengap"})],
        "drop_leading_mode": ["boolean"],
        "n_selection_resamples": [Interval(Integral, 1, None, closed="left")],
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "backend": [StrOptions({"random_forest", "extra_trees"})],
        "synthetic_method": [StrOptions({"bootstrap", "permutation", "uniform"})],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_clusters: int | Literal["auto"] = "auto",
        *,
        max_clusters: int = 20,
        cluster_selection: Literal["eigengap"] = "eigengap",
        drop_leading_mode: bool = True,
        n_selection_resamples: int = 3,
        n_estimators: int = 100,
        backend: Backend = "random_forest",
        synthetic_method: SamplingMethod = "bootstrap",
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.cluster_selection = cluster_selection
        self.drop_leading_mode = drop_leading_mode
        self.n_selection_resamples = n_selection_resamples
        self.n_estimators = n_estimators
        self.backend = backend
        self.synthetic_method = synthetic_method
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def fit(
        self,
        X: np.ndarray | sp.spmatrix,
        y: Any = None,
    ) -> DiscriminativeForestClusterer:
        """Fit the discriminator forest, select ``k``, and assign labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Present for API consistency with supervised estimators.

        Returns
        -------
        self : DiscriminativeForestClusterer
            Fitted estimator.
        """
        self._validate_params()
        X = validate_data(self, X, accept_sparse=["csc"])
        self.random_state_ = check_random_state(self.random_state)

        n_samples = X.shape[0]
        target_k = self._target_k(n_samples)
        n_components = self._n_components_for(target_k, n_samples)

        primary_spectrum = self._fit_primary_spectrum(X, n_components)

        if self.n_clusters == "auto":
            spectra = [primary_spectrum]
            for _ in range(self.n_selection_resamples - 1):
                Z_extra, n_extra = self._fit_extra_embedding(X)
                spectrum_extra = compute_leaf_spectrum(
                    Z_extra,
                    n_components=min(n_components, *Z_extra.shape),
                    n_estimators=n_extra,
                    random_state=self.random_state_,
                )
                spectra.append(spectrum_extra)

            selector = SpectralClusterCountSelector(
                max_clusters=self.max_clusters,
                drop_leading_mode=self.drop_leading_mode,
                strategy=self.cluster_selection,
            )
            self.cluster_selection_ = selector.select(spectra)
            self.n_clusters_ = int(self.cluster_selection_.n_clusters)
        else:
            self.n_clusters_ = int(self.n_clusters)
            self.cluster_selection_ = self._explicit_selection_result(
                primary_spectrum, self.n_clusters_
            )

        self.spectral_embedding_ = self._build_spectral_embedding(
            primary_spectrum, self.n_clusters_
        )

        kmeans_seed = int(self.random_state_.randint(np.iinfo(np.int32).max))
        kmeans = KMeans(
            n_clusters=self.n_clusters_,
            n_init=10,
            random_state=kmeans_seed,
        )
        self.labels_ = kmeans.fit_predict(self.spectral_embedding_)
        return self

    def fit_predict(
        self,
        X: np.ndarray | sp.spmatrix,
        y: Any = None,
    ) -> np.ndarray:
        """Fit the clusterer and return the cluster labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Present for API consistency with supervised estimators.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster index assigned to each sample.
        """
        self.fit(X, y)
        check_is_fitted(self, "labels_")
        return self.labels_

    def _target_k(self, n_samples: int) -> int:
        if self.n_clusters == "auto":
            return int(self.max_clusters)
        return int(self.n_clusters)

    def _n_components_for(self, target_k: int, n_samples: int) -> int:
        n_components = target_k + 5
        if self.drop_leading_mode:
            n_components += 1
        n_components = min(n_components, max(n_samples - 1, 1))
        return max(n_components, 2)

    def _fit_primary_spectrum(
        self,
        X: np.ndarray | sp.spmatrix,
        n_components: int,
    ) -> LeafSpectrum:
        forest_seed = int(self.random_state_.randint(np.iinfo(np.int32).max))
        embedding = DiscriminativeForestEmbedding(
            n_estimators=self.n_estimators,
            backend=self.backend,
            synthetic_method=self.synthetic_method,
            sparse_output=True,
            random_state=forest_seed,
        )
        Z = embedding.fit_transform(X)
        self.forest_embedding_ = embedding
        return compute_leaf_spectrum(
            Z,
            n_components=min(n_components, *Z.shape),
            n_estimators=self.n_estimators,
            random_state=self.random_state_,
        )

    def _fit_extra_embedding(
        self,
        X: np.ndarray | sp.spmatrix,
    ) -> tuple[sp.spmatrix, int]:
        seed = int(self.random_state_.randint(np.iinfo(np.int32).max))
        embedding = DiscriminativeForestEmbedding(
            n_estimators=self.n_estimators,
            backend=self.backend,
            synthetic_method=self.synthetic_method,
            sparse_output=True,
            random_state=seed,
        )
        Z = embedding.fit_transform(X)
        return Z, self.n_estimators

    def _build_spectral_embedding(
        self,
        spectrum: LeafSpectrum,
        n_clusters: int,
    ) -> np.ndarray:
        n_available = spectrum.eigenvectors.shape[1]
        start = 1 if (self.drop_leading_mode and n_available > n_clusters) else 0
        end = min(start + n_clusters, n_available)
        if end - start < 1:
            start = 0
            end = min(n_clusters, n_available)
        U = spectrum.eigenvectors[:, start:end]

        norms = np.linalg.norm(U, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0, norms, 1.0)
        return U / safe_norms

    def _explicit_selection_result(
        self,
        spectrum: LeafSpectrum,
        n_clusters: int,
    ) -> ClusterSelectionResult:
        eigengaps = _relative_eigengaps(spectrum.eigenvalues)
        return ClusterSelectionResult(
            n_clusters=n_clusters,
            eigenvalues=spectrum.eigenvalues.copy(),
            eigengaps=eigengaps,
            effective_rank=effective_rank(spectrum.eigenvalues),
            localization=inverse_participation_ratios(spectrum.eigenvectors),
            proposed_k_per_seed=np.array([n_clusters], dtype=np.int64),
            strategy="explicit",
            confidence="high",
        )
