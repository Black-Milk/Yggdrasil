"""Discriminator-trained random-forest embedding.

The :class:`DiscriminativeForestEmbedding` estimator fits a forest to the
binary task of distinguishing the input samples (class ``1``) from synthetic
samples (class ``0``) drawn from the empirical marginal distribution of the
input. Once fit, samples are embedded by the leaves they reach across the
ensemble.

The forest backend is pluggable: ``"random_forest"`` uses
:class:`~sklearn.ensemble.RandomForestClassifier` and ``"extra_trees"`` uses
:class:`~sklearn.ensemble.ExtraTreesClassifier`. The latter chooses split
thresholds at random within the selected feature, which typically increases
ensemble diversity and can improve the leaf embedding for downstream
clustering at the cost of higher per-tree bias.

This is conceptually distinct from :class:`sklearn.ensemble.RandomTreesEmbedding`,
which fits totally-random trees rather than a discriminator.

References
----------
Shi, T., & Horvath, S. (2006). "Unsupervised learning with random forest
predictors." *Journal of Computational and Graphical Statistics*, 15(1),
118-138.

Geurts, P., Ernst, D., & Wehenkel, L. (2006). "Extremely randomized trees."
*Machine Learning*, 63(1), 3-42.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data

from yggdrasil.utils.synthetic import SamplingMethod, generate_synthetic_features

__all__ = ["DiscriminativeForestEmbedding"]

Backend = Literal["random_forest", "extra_trees"]


class DiscriminativeForestEmbedding(TransformerMixin, BaseEstimator):
    """Unsupervised forest embedding via a real-vs-synthetic discriminator.

    Each call to :meth:`fit` constructs a synthetic counterpart to ``X`` from
    its empirical marginals, labels real rows as ``1`` and synthetic rows as
    ``0``, and trains an internal forest classifier on the resulting binary
    task. :meth:`transform` then returns the leaves each input sample reaches
    across the ensemble, optionally one-hot encoded into a sparse matrix.

    The forest backend is selected by the ``backend`` parameter. Use
    ``"random_forest"`` for the classical Shi-Horvath construction with
    greedy splits, or ``"extra_trees"`` for Extremely Randomized Trees,
    whose random split thresholds usually yield more diverse leaf
    partitions and therefore richer downstream similarity measures.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees in the forest.
    backend : {"random_forest", "extra_trees"}, default="random_forest"
        Which sklearn forest implementation to wrap. ``"random_forest"``
        uses :class:`~sklearn.ensemble.RandomForestClassifier` (greedy
        threshold search); ``"extra_trees"`` uses
        :class:`~sklearn.ensemble.ExtraTreesClassifier` (random thresholds).
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        Split criterion forwarded to the underlying forest.
    max_depth : int, optional, default=5
        Maximum depth of each tree.
    min_samples_split : int or float, default=2
        Minimum samples required to split an internal node.
    min_samples_leaf : int or float, default=1
        Minimum samples required at a leaf.
    min_weight_fraction_leaf : float, default=0.0
        Minimum weighted fraction of the input samples required to be at a
        leaf node.
    max_features : {"sqrt", "log2"}, int, float, or None, default="sqrt"
        Number of features considered when looking for the best split.
    max_leaf_nodes : int, optional
        Maximum number of leaf nodes per tree.
    bootstrap : bool, default=True
        Whether to bootstrap-sample when fitting individual trees. Note
        that :class:`~sklearn.ensemble.ExtraTreesClassifier` defaults to
        ``False`` upstream; this estimator defaults to ``True`` for both
        backends so they are directly comparable.
    oob_score : bool, default=False
        If ``True``, the underlying forest computes an out-of-bag
        decision function during :meth:`fit`, exposed as
        ``forest_.oob_decision_function_``. This is required by
        :func:`yggdrasil.clustering.diagnostics.discriminator_oob_auc`,
        which the discriminative-forest clusterer uses as a kernel
        informativeness gate. Requires ``bootstrap=True``.
    sparse_output : bool, default=True
        If ``True``, :meth:`transform` returns a sparse one-hot encoding of
        the leaves; otherwise it returns the raw leaf-index matrix of shape
        ``(n_samples, n_estimators)``.
    synthetic_method : {"bootstrap", "permutation", "uniform"}, default="bootstrap"
        How to draw the synthetic negative class. ``"bootstrap"`` resamples
        each column with replacement; ``"permutation"`` independently shuffles
        each column without replacement; ``"uniform"`` draws each column
        uniformly between its observed min and max.
    n_jobs : int, optional
        Number of parallel jobs for fitting and prediction.
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of synthetic-data generation and
        forest fitting. Pass an int for reproducible results across
        multiple calls. See :term:`Glossary <random_state>`.
    verbose : int, default=0
        Verbosity level forwarded to the underlying forest.
    warm_start : bool, default=False
        If ``True``, reuses the existing forest and adds more estimators
        instead of fitting a new one.

    Attributes
    ----------
    forest_ : RandomForestClassifier or ExtraTreesClassifier
        The fitted discriminator forest. Its concrete type matches the
        ``backend`` parameter.
    one_hot_encoder_ : OneHotEncoder
        Encoder used to one-hot encode leaf indices when
        ``sparse_output=True``.
    y_disc_ : ndarray of shape (2 * n_samples,)
        Binary real-vs-synthetic targets used to fit ``forest_``, in
        the same row order as ``forest_.oob_decision_function_``.
        Real samples are labeled ``1`` and synthetic samples ``0``.
        Useful for computing OOB metrics on the discriminator.
    random_state_ : RandomState
        The random number generator instantiated from ``random_state``.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during :meth:`fit`. Defined only when ``X``
        has feature names that are all strings.

    Examples
    --------
    >>> import numpy as np
    >>> from yggdrasil.clustering import DiscriminativeForestEmbedding
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(20, 4))
    >>> emb = DiscriminativeForestEmbedding(
    ...     n_estimators=5, backend="extra_trees", random_state=0
    ... ).fit(X)
    >>> emb.transform(X).shape[0]
    20
    """

    _BACKENDS = {
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,
    }

    def __init__(
        self,
        n_estimators: int = 10,
        *,
        backend: Backend = "random_forest",
        criterion: str = "gini",
        max_depth: int | None = 5,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: str | int | float | None = "sqrt",
        max_leaf_nodes: int | None = None,
        bootstrap: bool = True,
        oob_score: bool = False,
        sparse_output: bool = True,
        synthetic_method: SamplingMethod = "bootstrap",
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
        verbose: int = 0,
        warm_start: bool = False,
    ) -> None:
        self.n_estimators = n_estimators
        self.backend = backend
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.sparse_output = sparse_output
        self.synthetic_method = synthetic_method
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def _make_forest(self, random_state: int | np.random.RandomState | None) -> Any:
        """Instantiate the configured backend forest classifier."""
        try:
            forest_cls = self._BACKENDS[self.backend]
        except KeyError as exc:
            raise ValueError(
                f"backend must be one of {sorted(self._BACKENDS)}; got {self.backend!r}."
            ) from exc
        return forest_cls(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
        )

    def fit(
        self,
        X: np.ndarray | sp.spmatrix,
        y: Any = None,
        sample_weight: np.ndarray | None = None,
    ) -> DiscriminativeForestEmbedding:
        """Fit the embedding.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Present for API consistency with supervised estimators.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights for the real rows of ``X``. Synthetic rows are
            always given uniform weight ``1.0`` because they describe a
            fixed empirical-marginal contrast that is not under user
            control. Pass ``None`` to use uniform weights everywhere.

        Returns
        -------
        self : DiscriminativeForestEmbedding
            Fitted estimator.
        """
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

    def fit_transform(
        self,
        X: np.ndarray | sp.spmatrix,
        y: Any = None,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray | sp.spmatrix:
        """Fit the embedding and transform ``X`` in a single pass.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Present for API consistency with supervised estimators.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights for the real rows of ``X``. Synthetic rows
            are weighted at ``1.0``; see :meth:`fit` for the rationale.

        Returns
        -------
        X_embedded : {ndarray, sparse matrix} of shape (n_samples, n_out)
            Leaf embedding of ``X``. ``n_out = n_estimators`` when
            ``sparse_output=False``; otherwise ``n_out`` equals the total
            number of leaves across the ensemble.
        """
        X = validate_data(self, X, accept_sparse=["csc"])

        if sp.issparse(X):
            X.sort_indices()
            X_dense = X.toarray()
        else:
            X_dense = X

        self.random_state_ = check_random_state(self.random_state)
        n_samples = X_dense.shape[0]

        synth_X = generate_synthetic_features(
            X_dense, method=self.synthetic_method, random_state=self.random_state_
        )
        X_disc = np.vstack((X_dense, synth_X))
        y_disc = np.concatenate((np.ones(n_samples), np.zeros(n_samples)))

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X_dense)
            sw_disc = np.concatenate((sample_weight, np.ones(n_samples, dtype=sample_weight.dtype)))
        else:
            sw_disc = None

        # Permute real and synthetic rows together so the discriminator does
        # not see them in separate contiguous blocks; the same permutation is
        # applied to ``y_disc`` (and ``sw_disc`` when present) to keep them
        # aligned with one another and with ``forest_.oob_decision_function_``.
        perm = self.random_state_.permutation(2 * n_samples)
        X_disc = X_disc[perm]
        y_disc = y_disc[perm]
        if sw_disc is not None:
            sw_disc = sw_disc[perm]

        forest_seed = int(self.random_state_.randint(np.iinfo(np.int32).max))
        self.forest_ = self._make_forest(random_state=forest_seed)
        self.forest_.fit(X_disc, y_disc, sample_weight=sw_disc)
        self.y_disc_ = y_disc

        leaves = self.forest_.apply(X)
        if self.sparse_output:
            # Fit the encoder on the leaves visited by the *training* data
            # (real + synthetic), so every leaf the trained forest can return
            # is a known category at transform time. Fitting only on
            # ``apply(X)`` would miss leaves visited only by synthetic rows
            # and reject held-out real samples that later land in them.
            self.one_hot_encoder_ = OneHotEncoder(sparse_output=True)
            self.one_hot_encoder_.fit(self.forest_.apply(X_disc))
            return self.one_hot_encoder_.transform(leaves)
        return leaves

    def transform(self, X: np.ndarray | sp.spmatrix) -> np.ndarray | sp.spmatrix:
        """Embed ``X`` using the fitted forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to embed.

        Returns
        -------
        X_embedded : {ndarray, sparse matrix} of shape (n_samples, n_out)
            Leaf embedding of ``X``. ``n_out = n_estimators`` when
            ``sparse_output=False``; otherwise ``n_out`` equals the total
            number of leaves across the ensemble.
        """
        check_is_fitted(self, "forest_")
        X = validate_data(self, X, accept_sparse=["csc"], reset=False)
        leaves = self.forest_.apply(X)
        if self.sparse_output:
            return self.one_hot_encoder_.transform(leaves)
        return leaves
