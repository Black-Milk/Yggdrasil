"""Auto-:math:`k` clustering on discriminative-forest leaf kernels.

:class:`DiscriminativeForestClusterer` wraps the existing real-vs-synthetic
:class:`~yggdrasil.clustering.DiscriminativeForestEmbedding` with a
multi-signal cluster-count selector and a spectral-embedding ``k``-means
label assignment. For each reseeded forest it computes the leaf-kernel
spectrum once via truncated SVD on the sparse leaf-indicator matrix,
shares that spectrum between diagnostics and the embedding, computes
the discriminator OOB-AUC, and (in composite mode) scores each
candidate ``k`` via silhouette, label stability across reseeds,
Zelnik-Manor and Perona rotation cost, and Newman modularity. The
selector falls back to ``k = 2`` rather than raising on noisy spectra.

The synthetic samples used by the discriminator forest are training
contrast only; they are never clustered. After fitting the forest, the
clusterer applies it back to the original real rows and operates on
the resulting leaf embedding.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_random_state, validate_data

from yggdrasil.clustering.diagnostics.forest_quality import discriminator_oob_auc
from yggdrasil.clustering.diagnostics.spectrum import (
    LeafSpectrum,
    compute_leaf_spectrum,
    effective_rank,
    eigengap_curve,
    inverse_participation_ratios,
)
from yggdrasil.clustering.forest import Backend, DiscriminativeForestEmbedding
from yggdrasil.clustering.kernel import leaf_kernel
from yggdrasil.clustering.selector import (
    CandidateInputs,
    ClusterSelectionResult,
    SpectralClusterCountSelector,
)
from yggdrasil.utils.synthetic import SamplingMethod

__all__ = ["DiscriminativeForestClusterer"]


class DiscriminativeForestClusterer(ClusterMixin, BaseEstimator):
    """Clustering via discriminative forest leaf kernel with auto-:math:`k`.

    Trains an internal :class:`DiscriminativeForestEmbedding` to separate
    real samples from synthetic samples drawn from their empirical
    marginals, builds the leaf-kernel spectrum from the sparse
    leaf-indicator matrix of the real rows, selects ``n_clusters_``
    from that spectrum (or accepts a user-provided integer), and
    assigns labels by running k-means on the top eigenvectors of the
    kernel.

    By default the cluster-count selector runs in composite mode: a
    discriminator out-of-bag AUC gate fires first, then a candidate
    set is built from per-reseed eigengaps and cumulative spectral
    mass, intersected with effective rank, and each candidate is
    scored by a z-scored weighted sum of silhouette, label stability,
    Zelnik-Manor and Perona rotation cost, and Newman modularity.
    Setting ``cluster_selection="eigengap"`` reverts to the v1
    single-signal eigengap rule.

    Parameters
    ----------
    n_clusters : int or "auto", default="auto"
        Number of clusters. If ``"auto"``, ``n_clusters_`` is selected
        from the leaf-kernel diagnostics.
    max_clusters : int, default=20
        Largest cluster count considered when ``n_clusters="auto"``.
        Ignored when ``n_clusters`` is an explicit integer.
    cluster_selection : {"composite", "eigengap"}, default="composite"
        Cluster-count selection strategy. ``"composite"`` activates the
        full multi-signal selector; ``"eigengap"`` reproduces the v1
        single-signal eigengap rule bit-identically.
    drop_leading_mode : bool, default=True
        If ``True``, the largest eigenvalue/eigenvector pair is
        excluded when scoring eigengaps and when forming the spectral
        embedding. The leading mode of a forest proximity kernel is
        usually a global "everything connected" component that
        obscures cluster structure.
    n_selection_resamples : int, default=3
        Number of forests fit during ``n_clusters="auto"`` selection,
        each with a different seed. The composite selector uses these
        for label-stability scoring; the eigengap selector uses them
        for modal-:math:`k` voting.
    n_estimators : int, default=100
        Number of trees in each fitted forest.
    backend : {"random_forest", "extra_trees"}, default="random_forest"
        Forest implementation forwarded to
        :class:`DiscriminativeForestEmbedding`.
    synthetic_method : {"bootstrap", "permutation", "uniform"}, default="bootstrap"
        Synthetic-sampling method used by the discriminator forest.
    min_discriminator_auc : float, default=0.55
        Composite-mode AUC gate. When the mean discriminator OOB-AUC
        across reseeds is below this threshold the selector returns
        ``n_clusters_=2`` with ``confidence="low"`` and
        ``gating_reason="discriminator_auc_below_threshold"``.
        Ignored when ``cluster_selection="eigengap"``.
    cumulative_mass_threshold : float, default=0.9
        Threshold for the cumulative-spectral-mass candidate.
    silhouette_weight : float, default=1.0
        Composite weight on the silhouette signal.
    stability_weight : float, default=1.0
        Composite weight on the label-stability signal.
    rotation_weight : float, default=1.0
        Composite weight on the (negated) rotation-cost signal.
    modularity_weight : float, default=0.0
        Composite weight on Newman modularity. Off by default because
        modularity requires materializing dense ``O(n_samples^2)``
        kernels.
    localization_threshold : float or None, default=None
        Optional inverse-participation-ratio cutoff used to drop
        candidate ``k`` whose top-:math:`k` eigenvectors are too
        concentrated on a few samples.
    confidence_margin : float, default=0.5
        Composite-score margin (in z-score units) required between the
        winner and runner-up for ``confidence="high"``.
    stability_floor : float, default=0.6
        Minimum mean ARI across reseeds at the winning ``k`` required
        for ``confidence="high"``.
    random_state : int, RandomState instance or None, default=None
        Controls all randomness: synthetic-sample generation, forest
        seeds, the SVD initialization, the rotation-cost optimizer,
        and the per-(k, reseed) k-means runs.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster index assigned to each training sample.
    n_clusters_ : int
        Number of clusters used for the final assignment.
    spectral_embedding_ : ndarray of shape (n_samples, n_components)
        Row-normalized matrix of top eigenvectors used as the input to
        k-means. ``n_components <= n_clusters_`` (one less when
        ``drop_leading_mode=True`` and the leading mode was excluded).
    forest_embedding_ : DiscriminativeForestEmbedding
        The fitted underlying embedding for the primary reseed.
    cluster_selection_ : ClusterSelectionResult
        Structured summary of how ``n_clusters_`` was chosen,
        including eigenvalues, eigengaps, effective rank, per-seed
        proposals, and (in composite mode) per-:math:`k` silhouette,
        stability, rotation cost, modularity, composite scores, and
        the discriminator OOB-AUC.
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
    computed once via truncated SVD on the sparse leaf-indicator
    matrix; the dense :math:`n \\times n` kernel is never materialized
    unless ``modularity_weight > 0``.

    Kernel ``k``-means against the precomputed leaf kernel is
    documented as an alternative in
    ``docs/random_forest_leaf_kernel_recipes.md`` but is not provided
    as a backend in this version.

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
        "cluster_selection": [StrOptions({"eigengap", "composite"})],
        "drop_leading_mode": ["boolean"],
        "n_selection_resamples": [Interval(Integral, 1, None, closed="left")],
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "backend": [StrOptions({"random_forest", "extra_trees"})],
        "synthetic_method": [StrOptions({"bootstrap", "permutation", "uniform"})],
        "min_discriminator_auc": [Interval(Real, 0.0, 1.0, closed="both")],
        "cumulative_mass_threshold": [Interval(Real, 0.0, 1.0, closed="right")],
        "silhouette_weight": [Interval(Real, 0.0, None, closed="left")],
        "stability_weight": [Interval(Real, 0.0, None, closed="left")],
        "rotation_weight": [Interval(Real, 0.0, None, closed="left")],
        "modularity_weight": [Interval(Real, 0.0, None, closed="left")],
        "localization_threshold": [Interval(Real, 0.0, 1.0, closed="both"), None],
        "confidence_margin": [Interval(Real, 0.0, None, closed="left")],
        "stability_floor": [Interval(Real, -1.0, 1.0, closed="both")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_clusters: int | Literal["auto"] = "auto",
        *,
        max_clusters: int = 20,
        cluster_selection: Literal["eigengap", "composite"] = "composite",
        drop_leading_mode: bool = True,
        n_selection_resamples: int = 3,
        n_estimators: int = 100,
        backend: Backend = "random_forest",
        synthetic_method: SamplingMethod = "bootstrap",
        min_discriminator_auc: float = 0.55,
        cumulative_mass_threshold: float = 0.9,
        silhouette_weight: float = 1.0,
        stability_weight: float = 1.0,
        rotation_weight: float = 1.0,
        modularity_weight: float = 0.0,
        localization_threshold: float | None = None,
        confidence_margin: float = 0.5,
        stability_floor: float = 0.6,
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
        self.min_discriminator_auc = min_discriminator_auc
        self.cumulative_mass_threshold = cumulative_mass_threshold
        self.silhouette_weight = silhouette_weight
        self.stability_weight = stability_weight
        self.rotation_weight = rotation_weight
        self.modularity_weight = modularity_weight
        self.localization_threshold = localization_threshold
        self.confidence_margin = confidence_margin
        self.stability_floor = stability_floor
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

        keep_z = self.modularity_weight > 0.0
        primary_spectrum, primary_embedding, primary_y_disc, primary_Z = self._fit_primary_spectrum(
            X, n_components, keep_z=keep_z
        )
        self.forest_embedding_ = primary_embedding
        primary_auc = self._safe_oob_auc(primary_embedding, primary_y_disc)

        if self.n_clusters == "auto":
            spectra: list[LeafSpectrum] = [primary_spectrum]
            cached_zs: list[sp.csr_matrix | None] = [primary_Z]
            extra_aucs: list[float] = []
            for _ in range(self.n_selection_resamples - 1):
                spectrum_extra, auc_extra, Z_extra = self._fit_extra_spectrum(
                    X, n_components, keep_z=keep_z
                )
                spectra.append(spectrum_extra)
                cached_zs.append(Z_extra)
                if auc_extra is not None:
                    extra_aucs.append(auc_extra)

            auc_values = [primary_auc, *extra_aucs] if primary_auc is not None else extra_aucs
            mean_auc = float(np.mean(auc_values)) if auc_values else None

            selector = SpectralClusterCountSelector(
                max_clusters=self.max_clusters,
                drop_leading_mode=self.drop_leading_mode,
                strategy=self.cluster_selection,
                min_discriminator_auc=self.min_discriminator_auc,
                cumulative_mass_threshold=self.cumulative_mass_threshold,
                silhouette_weight=self.silhouette_weight,
                stability_weight=self.stability_weight,
                rotation_weight=self.rotation_weight,
                modularity_weight=self.modularity_weight,
                localization_threshold=self.localization_threshold,
                confidence_margin=self.confidence_margin,
                stability_floor=self.stability_floor,
            )

            candidate_inputs: CandidateInputs | None = None
            embeddings_per_k_per_seed: dict[int, list[np.ndarray]] = {}
            labelings_per_k_per_seed: dict[int, list[np.ndarray]] = {}
            if self.cluster_selection == "composite":
                candidates = selector.candidate_set(spectra)
                kernels = self._maybe_build_kernels(cached_zs) if candidates else None
                for k in candidates:
                    embeddings_per_k_per_seed[k] = []
                    labelings_per_k_per_seed[k] = []
                    for spectrum in spectra:
                        embedding = self._build_spectral_embedding(spectrum, k)
                        labels = self._kmeans_labels(embedding, k)
                        embeddings_per_k_per_seed[k].append(embedding)
                        labelings_per_k_per_seed[k].append(labels)
                candidate_inputs = CandidateInputs(
                    labelings=labelings_per_k_per_seed,
                    embeddings=embeddings_per_k_per_seed,
                    kernels=kernels,
                )

            self.cluster_selection_ = selector.select(
                spectra,
                candidate_inputs=candidate_inputs,
                discriminator_auc=mean_auc,
            )
            self.n_clusters_ = int(self.cluster_selection_.n_clusters)

            cached_embedding = (
                embeddings_per_k_per_seed.get(self.n_clusters_, [None])[0]
                if self.cluster_selection == "composite"
                else None
            )
            cached_labels = (
                labelings_per_k_per_seed.get(self.n_clusters_, [None])[0]
                if self.cluster_selection == "composite"
                else None
            )
        else:
            self.n_clusters_ = int(self.n_clusters)
            self.cluster_selection_ = self._explicit_selection_result(
                primary_spectrum, self.n_clusters_, primary_auc
            )
            cached_embedding = None
            cached_labels = None

        if cached_embedding is not None:
            self.spectral_embedding_ = cached_embedding
        else:
            self.spectral_embedding_ = self._build_spectral_embedding(
                primary_spectrum, self.n_clusters_
            )

        if cached_labels is not None:
            self.labels_ = cached_labels.astype(np.int64, copy=True)
        else:
            self.labels_ = self._kmeans_labels(self.spectral_embedding_, self.n_clusters_)
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

    def _make_embedding(self, seed: int) -> DiscriminativeForestEmbedding:
        return DiscriminativeForestEmbedding(
            n_estimators=self.n_estimators,
            backend=self.backend,
            synthetic_method=self.synthetic_method,
            sparse_output=True,
            oob_score=True,
            random_state=seed,
        )

    def _fit_primary_spectrum(
        self,
        X: np.ndarray | sp.spmatrix,
        n_components: int,
        *,
        keep_z: bool,
    ) -> tuple[
        LeafSpectrum,
        DiscriminativeForestEmbedding,
        np.ndarray,
        sp.csr_matrix | None,
    ]:
        seed = int(self.random_state_.randint(np.iinfo(np.int32).max))
        embedding = self._make_embedding(seed)
        Z = embedding.fit_transform(X)
        spectrum = compute_leaf_spectrum(
            Z,
            n_components=min(n_components, *Z.shape),
            n_estimators=self.n_estimators,
            random_state=self.random_state_,
        )
        cached_Z = sp.csr_matrix(Z) if keep_z else None
        return spectrum, embedding, embedding.y_disc_, cached_Z

    def _fit_extra_spectrum(
        self,
        X: np.ndarray | sp.spmatrix,
        n_components: int,
        *,
        keep_z: bool,
    ) -> tuple[LeafSpectrum, float | None, sp.csr_matrix | None]:
        seed = int(self.random_state_.randint(np.iinfo(np.int32).max))
        embedding = self._make_embedding(seed)
        Z = embedding.fit_transform(X)
        spectrum = compute_leaf_spectrum(
            Z,
            n_components=min(n_components, *Z.shape),
            n_estimators=self.n_estimators,
            random_state=self.random_state_,
        )
        auc = self._safe_oob_auc(embedding, embedding.y_disc_)
        cached_Z = sp.csr_matrix(Z) if keep_z else None
        return spectrum, auc, cached_Z

    def _safe_oob_auc(
        self,
        embedding: DiscriminativeForestEmbedding,
        y_disc: np.ndarray,
    ) -> float | None:
        try:
            return float(discriminator_oob_auc(embedding.forest_, y_disc))
        except AttributeError, ValueError:
            return None

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

    def _kmeans_labels(self, embedding: np.ndarray, n_clusters: int) -> np.ndarray:
        kmeans_seed = int(self.random_state_.randint(np.iinfo(np.int32).max))
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=kmeans_seed,
        )
        labels = kmeans.fit_predict(embedding)
        return labels.astype(np.int64, copy=False)

    def _maybe_build_kernels(
        self,
        cached_zs: list[sp.csr_matrix | None],
    ) -> list[np.ndarray] | None:
        """Materialize the exact dense kernel for each reseed if modularity is on.

        Computes ``K = leaf_kernel(Z, n_estimators)`` rather than the
        truncated SVD reconstruction ``U Λ Uᵀ``. The truncated form
        underestimates trace and degree, can have negative entries, and
        biases :func:`modularity_on_kernel`; the exact form is
        non-negative, has unit diagonal, and matches what the rest of
        the leaf-kernel toolbox documents in
        ``docs/random_forest_leaf_kernel_recipes.md``.
        """
        if self.modularity_weight <= 0.0:
            return None
        kernels: list[np.ndarray] = []
        for Z in cached_zs:
            if Z is None:
                continue
            kernels.append(leaf_kernel(Z, n_estimators=self.n_estimators))
        return kernels

    def _explicit_selection_result(
        self,
        spectrum: LeafSpectrum,
        n_clusters: int,
        discriminator_auc: float | None,
    ) -> ClusterSelectionResult:
        return ClusterSelectionResult(
            n_clusters=n_clusters,
            eigenvalues=spectrum.eigenvalues.copy(),
            eigengaps=eigengap_curve(spectrum.eigenvalues),
            effective_rank=effective_rank(spectrum.eigenvalues),
            localization=inverse_participation_ratios(spectrum.eigenvectors),
            proposed_k_per_seed=np.array([n_clusters], dtype=np.int64),
            strategy="explicit",
            confidence="high",
            discriminator_auc=discriminator_auc,
        )
