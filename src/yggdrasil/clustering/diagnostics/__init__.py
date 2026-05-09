"""Spectral and partition diagnostics for forest leaf kernels.

This sub-package collects the read-only signals consumed by
:class:`yggdrasil.clustering.SpectralClusterCountSelector`:

- :mod:`.spectrum` — leaf-kernel spectrum, effective rank, cumulative
  spectral mass, eigengap curve, and inverse participation ratios.
- :mod:`.partition` — partition-quality metrics that act on labelings
  (silhouette on a spectral embedding, Newman modularity on a kernel).
- :mod:`.stability` — label-stability metrics across resampled fits.
- :mod:`.rotation` — Zelnik-Manor and Perona rotation cost on top
  eigenvectors.
- :mod:`.forest_quality` — discriminator out-of-bag AUC and an
  informativeness predicate.

The cluster-count selector itself (and the
:class:`~yggdrasil.clustering.ClusterSelectionResult` it returns) lives
in :mod:`yggdrasil.clustering.selector`. For backwards compatibility
with the v1 module, those symbols are re-exported here so
``from yggdrasil.clustering.diagnostics import ClusterSelectionResult``
keeps working.
"""

from yggdrasil.clustering.diagnostics.forest_quality import (
    discriminator_oob_auc,
    is_kernel_informative,
)
from yggdrasil.clustering.diagnostics.partition import (
    modularity_on_kernel,
    silhouette_on_embedding,
)
from yggdrasil.clustering.diagnostics.rotation import rotation_cost
from yggdrasil.clustering.diagnostics.spectrum import (
    LeafSpectrum,
    compute_leaf_spectrum,
    cumulative_spectral_mass,
    effective_rank,
    eigengap_curve,
    inverse_participation_ratios,
)
from yggdrasil.clustering.diagnostics.stability import (
    label_stability,
    pairwise_ari_matrix,
)

__all__ = [
    "LeafSpectrum",
    "compute_leaf_spectrum",
    "cumulative_spectral_mass",
    "discriminator_oob_auc",
    "effective_rank",
    "eigengap_curve",
    "inverse_participation_ratios",
    "is_kernel_informative",
    "label_stability",
    "modularity_on_kernel",
    "pairwise_ari_matrix",
    "rotation_cost",
    "silhouette_on_embedding",
]


def __getattr__(name: str):
    """Lazily forward selector symbols to ``yggdrasil.clustering.selector``.

    The selector imports from this sub-package, so an eager re-export
    here would create an import cycle. Routing
    ``ClusterSelectionResult`` and ``SpectralClusterCountSelector``
    through ``__getattr__`` preserves the v1 import path
    ``from yggdrasil.clustering.diagnostics import ...`` without
    introducing a cycle.
    """
    if name in {"ClusterSelectionResult", "SpectralClusterCountSelector"}:
        from yggdrasil.clustering import selector as _selector

        return getattr(_selector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
