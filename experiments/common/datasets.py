"""Synthetic dataset generators for composite-scorer experiments.

Each generator returns ``(X, y_true, name)`` where:

- ``X`` is the feature matrix, shape ``(n_samples, n_features)``.
- ``y_true`` is the ground-truth label per row, shape ``(n_samples,)``.
- ``name`` is a short slug that identifies the dataset in result tables.

The "trap" generators take a clean base dataset and inject a structural
anomaly that v2's composite selector is suspected to mishandle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_blobs


@dataclass(frozen=True)
class LabeledDataset:
    """Bundle of features, ground-truth labels, and a slug name."""

    X: np.ndarray
    y: np.ndarray
    name: str
    metadata: dict

    @property
    def k_true(self) -> int:
        return int(np.unique(self.y).size)


def clean_blobs(
    *,
    n_centers: int,
    n_per_center: int = 40,
    n_features: int = 4,
    cluster_std: float = 0.5,
    random_state: int = 0,
    name: str | None = None,
) -> LabeledDataset:
    """Standard isotropic Gaussian blobs.

    ``cluster_std`` is the *raw* per-cluster standard deviation passed to
    :func:`sklearn.datasets.make_blobs`; centers are drawn from
    ``[-10, 10]^n_features`` per sklearn's default. Smaller ``cluster_std``
    relative to the inter-center spacing means more separable clusters.
    """
    n_samples = n_centers * n_per_center
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    auto_name = (
        name
        or f"clean_k{n_centers}_per{n_per_center}_d{n_features}_std{cluster_std:.2f}".replace(
            ".", "p"
        )
    )
    return LabeledDataset(
        X=X,
        y=y,
        name=auto_name,
        metadata={
            "kind": "clean_blobs",
            "n_centers": n_centers,
            "n_per_center": n_per_center,
            "n_features": n_features,
            "cluster_std": cluster_std,
            "random_state": random_state,
        },
    )


def add_near_duplicates(
    base: LabeledDataset,
    *,
    n_duplicates: int,
    target_cluster: int = 0,
    jitter: float = 1e-3,
    random_state: int = 0,
    name: str | None = None,
) -> LabeledDataset:
    """Inject ``n_duplicates`` near-duplicates of one row in ``target_cluster``.

    The duplicates are labeled as belonging to ``target_cluster`` (so the
    ground-truth ``k_true`` is unchanged), but they form a tight rank-1
    sub-cluster that the in-bag leaf kernel amplifies.
    """
    if n_duplicates <= 0:
        return base
    seed_idx = int(np.where(base.y == target_cluster)[0][0])
    rng = np.random.default_rng(random_state)
    dup_x = np.tile(base.X[seed_idx], (n_duplicates, 1)) + rng.normal(
        scale=jitter, size=(n_duplicates, base.X.shape[1])
    )
    X = np.vstack([base.X, dup_x])
    y = np.concatenate([base.y, np.full(n_duplicates, target_cluster, dtype=base.y.dtype)])
    return LabeledDataset(
        X=X,
        y=y,
        name=name or f"{base.name}+dup{n_duplicates}",
        metadata={
            **base.metadata,
            "trap": "near_duplicates",
            "n_duplicates": n_duplicates,
            "target_cluster": target_cluster,
            "jitter": jitter,
        },
    )


def add_tiny_dense_outlier_cluster(
    base: LabeledDataset,
    *,
    n_extra: int,
    location_scale: float = 5.0,
    cluster_std: float = 0.05,
    random_state: int = 0,
    name: str | None = None,
) -> LabeledDataset:
    """Inject a small, very tight, distant cluster of ``n_extra`` rows.

    The new rows are NOT counted toward ``k_true``: they are labeled as
    a synthetic out-of-distribution group (label = ``base.k_true``). This
    is exactly the trap pattern where the in-bag leaf kernel reports an
    extra mode that is structural noise rather than a real cluster.
    """
    if n_extra <= 0:
        return base
    rng = np.random.default_rng(random_state)
    center = rng.normal(loc=0.0, scale=location_scale, size=base.X.shape[1])
    extra = rng.normal(loc=center, scale=cluster_std, size=(n_extra, base.X.shape[1]))
    out_label = int(base.y.max() + 1)
    X = np.vstack([base.X, extra])
    y = np.concatenate([base.y, np.full(n_extra, out_label, dtype=base.y.dtype)])
    return LabeledDataset(
        X=X,
        y=y,
        name=name or f"{base.name}+tinydense{n_extra}",
        metadata={
            **base.metadata,
            "trap": "tiny_dense_outlier",
            "n_extra": n_extra,
            "location_scale": location_scale,
            "extra_cluster_std": cluster_std,
        },
    )


def add_shortcut_feature(
    base: LabeledDataset,
    *,
    cut: float | None = None,
    random_state: int = 0,
    name: str | None = None,
) -> LabeledDataset:
    """Append a binary "shortcut" feature that perfectly splits the data
    on an arbitrary threshold of the first existing feature.

    The shortcut doesn't change the ground-truth labels but offers the
    discriminator a near-perfect axis-aligned split that has nothing to
    do with the real cluster structure. A single-feature shortcut is
    catastrophic for proximity-based clustering when its leaves are
    near-pure under that split.
    """
    rng = np.random.default_rng(random_state)
    feat0 = base.X[:, 0]
    if cut is None:
        cut = float(np.median(feat0))
    shortcut = (feat0 > cut).astype(np.float64)
    shortcut += rng.normal(scale=1e-3, size=shortcut.shape)
    X = np.hstack([base.X, shortcut[:, None]])
    return LabeledDataset(
        X=X,
        y=base.y.copy(),
        name=name or f"{base.name}+shortcut",
        metadata={
            **base.metadata,
            "trap": "shortcut_feature",
            "cut": cut,
        },
    )


def imbalanced_blobs(
    *,
    sizes: tuple[int, ...],
    n_features: int = 4,
    cluster_std: float = 0.5,
    random_state: int = 0,
    name: str | None = None,
) -> LabeledDataset:
    """Imbalanced isotropic blobs: ``len(sizes)`` clusters with the given
    per-cluster row counts."""
    rng = np.random.default_rng(random_state)
    centers = rng.normal(scale=10.0, size=(len(sizes), n_features))
    X_list = []
    y_list = []
    for i, (size, center) in enumerate(zip(sizes, centers, strict=True)):
        X_list.append(rng.normal(loc=center, scale=cluster_std, size=(size, n_features)))
        y_list.append(np.full(size, i, dtype=np.int64))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return LabeledDataset(
        X=X,
        y=y,
        name=name or f"imbal_{'-'.join(str(s) for s in sizes)}_d{n_features}",
        metadata={
            "kind": "imbalanced_blobs",
            "sizes": list(sizes),
            "n_features": n_features,
            "cluster_std": cluster_std,
            "random_state": random_state,
        },
    )
