"""Tree-ensemble clustering and embedding estimators."""

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
from yggdrasil.clustering.discriminative import DiscriminativeForestClusterer
from yggdrasil.clustering.forest import DiscriminativeForestEmbedding
from yggdrasil.clustering.kernel import (
    kernel_from_leaves,
    leaf_dissimilarity,
    leaf_indicator_matrix,
    leaf_kernel,
)
from yggdrasil.clustering.selector import (
    CandidateInputs,
    ClusterSelectionResult,
    SpectralClusterCountSelector,
)

__all__ = [
    "CandidateInputs",
    "ClusterSelectionResult",
    "DiscriminativeForestClusterer",
    "DiscriminativeForestEmbedding",
    "LeafSpectrum",
    "SpectralClusterCountSelector",
    "compute_leaf_spectrum",
    "cumulative_spectral_mass",
    "discriminator_oob_auc",
    "effective_rank",
    "eigengap_curve",
    "inverse_participation_ratios",
    "is_kernel_informative",
    "kernel_from_leaves",
    "label_stability",
    "leaf_dissimilarity",
    "leaf_indicator_matrix",
    "leaf_kernel",
    "modularity_on_kernel",
    "pairwise_ari_matrix",
    "rotation_cost",
    "silhouette_on_embedding",
]
