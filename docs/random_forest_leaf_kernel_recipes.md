# Random Forest Leaf-Kernel Recipes

This note collects methodology recipes for adapting WeightWatcher-style spectral
diagnostics to random forests and random-forest-derived clustering. The emphasis
is conceptual: what matrix to build, what it means, how to interpret its
spectrum, and how it can guide clustering or regularization.

The central shift is from boosting dynamics to partition geometry.

For boosted trees, the diagnostic object can be built from out-of-fold margin
increments along the boosting trajectory. For random forests, there is no
equivalent sequential trajectory. Instead, the forest induces a data-dependent
geometry through terminal leaf co-membership.

## 1. Forest Leaf Proximity

For a random forest with trees $t = 1,\dots,T$, let

$$
\ell_t(x_i)
$$

denote the terminal leaf reached by sample $x_i$ in tree $t$.

The basic forest proximity kernel is

$$
K_{ij}
= \frac{1}{T}
\sum_{t=1}^{T}
\mathbf{1}\{\ell_t(x_i) = \ell_t(x_j)\}.
$$

Two samples are similar if many trees assign them to the same terminal region.
This turns the forest into a learned similarity machine.

The corresponding sparse leaf feature map is

$$
Z_{i,(t,\ell)}
= \mathbf{1}\{x_i \text{ lands in leaf } \ell \text{ of tree } t\}.
$$

Then

$$
K = \frac{1}{T} Z Z^\top.
$$

The matrix $Z$ is the explicit leaf-assignment operator. The matrix $K$ is
the sample-side Gram matrix induced by that operator.

## 2. OOB Proximity As The OOF Analogue

If the goal is a generalization-sensitive diagnostic, the kernel should avoid
using trees that trained on the samples being compared. Random forests make this
natural through out-of-bag samples.

Let $i \in OOB_t$ mean sample $i$ was not included in the bootstrap sample
for tree $t$. A strict OOB proximity kernel is

$$
K_{ij}^{OOB}
=
\frac{
\sum_t
\mathbf{1}\{i,j \in OOB_t\}
\mathbf{1}\{\ell_t(x_i)=\ell_t(x_j)\}
}{
\sum_t \mathbf{1}\{i,j \in OOB_t\}
}.
$$

This compares two samples only using trees for which both samples were out of
bag. It is the closest random-forest analogue to the out-of-fold construction
used for boosting diagnostics.

The main tradeoff is coverage. For bootstrap forests, a sample is OOB for a tree
with probability approximately $e^{-1} \approx 0.368$. A pair of samples is
jointly OOB with probability approximately $0.368^2 \approx 0.135$. Therefore,
pairwise OOB proximity can be noisy for small forests and becomes more stable as
the number of trees grows.

## 3. WeightWatcher Mapping

WeightWatcher-style diagnostics usually study the spectrum of a learned operator
or its correlation matrix. For random forests, the analogue is:

- learned operator: sparse leaf assignment matrix $Z$
- correlation or Gram matrix: $K = ZZ^\top / T$
- singular values: singular values of $Z$
- eigenvalues: eigenvalues of $K$, equal to squared singular values of $Z$
  up to normalization

The conceptual bridge is:

$$
\text{XGBoost: OOF margin-increment operator}
$$

$$
\text{Random forest: OOB leaf-assignment/proximity operator}
$$

Both are surrogate operators built from tree ensembles. Both can be analyzed
spectrally to detect structure beyond accuracy or OOB error.

## 4. Spectral Interpretations

The RF leaf-kernel spectrum describes the geometry of forest partitions.

A few useful signatures:

- A clear group of dominant eigenvalues suggests coherent large-scale clusters.
- A large eigengap after $k$ dominant eigenvalues suggests a candidate
  $k$-cluster structure.
- Many isolated spikes can indicate dominant subpopulations, duplicates,
  leakage groups, rare-category shortcuts, or partition traps.
- A heavy-tailed spectrum suggests multiscale partition geometry.
- A very noisy high-rank bulk can indicate fragmentation or sample isolation.
- A low-rank spectrum can indicate coarse underfit partitions.
- Instability across forest seeds can indicate unreliable structure.
- A large full-kernel versus OOB-kernel spectral gap can indicate overfitting.

Alpha should be interpreted as a global spectral texture, not as a direct
cluster count. Cluster count is better inferred from eigengaps, spike structure,
and stability across seeds.

## 5. Alpha Diagnostic

Alpha describes the slope of the heavy-tailed part of the empirical spectral
density. In this setting, the spectrum may come from the singular values of the
leaf feature matrix $Z$, or from the eigenvalues of the proximity kernel $K$.

For a power-law region of eigenvalues,

$$
\rho(\lambda) \sim \lambda^{-\alpha},
$$

alpha summarizes how quickly spectral mass decays. Lower alpha usually means a
heavier tail: more spectral mass persists across many scales. Higher alpha means
the spectrum decays more quickly and can look more weakly correlated or
random-like.

For random forest leaf kernels, alpha should be read as a diagnostic of
partition geometry:

- A moderate heavy tail suggests multiscale forest structure: the forest has
  discovered organization at several resolutions.
- Very high alpha can suggest weakly organized geometry, underfitting, or a
  spectrum close to noise.
- Very low alpha can indicate strong correlated structure, but it needs context:
  it may reflect useful hierarchy, or it may reflect shortcut features,
  duplicates, leakage, or dominant partition traps.
- Alpha instability across forest seeds suggests the geometry is not robust.

Alpha is not a cluster-count estimator. It says something about the global
texture of the spectrum, not the number of coherent groups. For clustering, use
alpha alongside eigengaps, spike count, and stability.

## 6. Eigengap Diagnostic

Eigengaps measure separation between consecutive eigenvalues of the proximity
kernel. If

$$
\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n,
$$

then a simple eigengap score is

$$
g_k = \lambda_k - \lambda_{k+1}.
$$

One can also use a relative gap,

$$
r_k = \frac{\lambda_k - \lambda_{k+1}}{\lambda_{k+1} + \epsilon}.
$$

For clustering, a large gap after $\lambda_k$ suggests that the forest sees
approximately $k$ dominant directions of variation. In a clean block-like
similarity matrix, this often corresponds to $k$ major clusters.

Interpretation:

- A large early eigengap suggests a small number of strong coarse clusters.
- Several meaningful gaps at different scales suggest hierarchical or
  multiresolution structure.
- No clear gaps, but a long heavy tail, suggests continuous or multiscale
  organization rather than a clean flat clustering.
- Gaps that disappear under OOB construction or across random seeds should be
  treated as unstable.

Eigengaps are most meaningful on a centered or normalized kernel, depending on
the clustering goal. Raw proximity matrices often have a dominant global
similarity mode, so the first eigenvalue may mostly reflect overall connectedness
rather than a useful cluster.

## 7. Spike Count Diagnostic

Spike count measures how many eigenvalues stand clearly above the spectral bulk.
In the RF proximity setting, spikes are candidate dominant structures in the
forest-induced similarity graph.

Conceptually, a spike is an eigenvalue that is too large to be explained by the
background bulk spectrum. The exact threshold can be chosen empirically using a
randomized baseline, seed variation, or a fitted bulk model.

Useful baselines include:

- a permuted-leaf baseline, where leaf assignments are shuffled within trees
- a label-permutation baseline for supervised forests
- a synthetic-data baseline for discriminative forest clustering
- a seed ensemble baseline, where spikes must persist across repeated forests

Interpretation:

- A small number of stable spikes can indicate strong, meaningful clusters.
- Many spikes may indicate fragmented structure, rare categories, duplicates, or
  many small isolated groups.
- A spike with a localized eigenvector can indicate a partition trap rather than
  a broad cluster.
- Spikes present in the full kernel but absent in the OOB kernel can indicate
  in-bag memorization.

Spike count is closely related to cluster discovery, but it should not be used
alone. Stable spikes with interpretable eigenvectors are more meaningful than
raw spike count.

## 8. Effective Rank Diagnostic

Effective rank measures how many spectral directions meaningfully contribute to
the kernel. It is smoother than a hard rank and more informative for noisy
spectra.

One common definition uses normalized eigenvalues

$$
p_i = \frac{\lambda_i}{\sum_j \lambda_j},
$$

and spectral entropy

$$
H = -\sum_i p_i \log p_i.
$$

The effective rank is then

$$
r_{\mathrm{eff}} = \exp(H).
$$

Another useful quantity is stable rank:

$$
r_{\mathrm{stable}} =
\frac{\sum_i \lambda_i}{\lambda_1}.
$$

For RF leaf kernels, effective rank summarizes the dimensionality of the
forest-induced partition geometry.

Interpretation:

- Very low effective rank suggests coarse partitions or dominance by a few
  global modes.
- Moderate effective rank suggests the forest uses several meaningful partition
  directions without excessive fragmentation.
- Very high effective rank can indicate deep, tiny-leaf fragmentation or
  sample-level memorization.
- Effective rank that is much higher in the full kernel than the OOB kernel can
  indicate in-bag structure that does not generalize.

Effective rank is especially useful in regularization sweeps. Increasing
`max_depth` or decreasing `min_samples_leaf` should usually increase effective
rank. If predictive performance stops improving while effective rank continues
to grow, the forest may be entering a memorization regime.

## 9. Top-Eigenvector Localization Diagnostic

Top-eigenvector localization asks whether a dominant spectral mode is spread
across many samples or concentrated on a small subset.

For an eigenvector $v$, a standard localization score is the inverse
participation ratio:

$$
IPR(v) = \sum_i v_i^4,
$$

assuming $v$ is normalized so that $\sum_i v_i^2 = 1$. A related effective
support size is

$$
n_{\mathrm{eff}}(v) = \frac{1}{IPR(v)}.
$$

If $n_{\mathrm{eff}}$ is large, the eigenvector is distributed across many
samples. If it is small, the eigenvector is localized on a small group.

Interpretation:

- A broad top eigenvector can represent a global cluster, class axis, or major
  population structure.
- A localized top eigenvector can reveal duplicates, rare categories, leakage,
  anomalies, or partition traps.
- Localized eigenvectors paired with large spikes deserve inspection: the spike
  may be driven by a tiny subset rather than a meaningful cluster.
- Stable localization across seeds suggests a real substructure; unstable
  localization suggests noise or overfitting.

This diagnostic is useful because eigenvalues alone do not tell whether a
spectral mode is broad or narrow. Two forests can have similar spike counts, but
one may have broad cluster modes while the other is dominated by tiny isolated
groups.

## 10. Alpha As Regularization Feedback

Alpha signatures can guide random forest regularization, but they should not be
used as a blind objective.

The practical loop is:

1. Train a forest candidate.
2. Build $Z$, $K$, or $K^{OOB}$.
3. Compute spectral diagnostics: alpha, spike count, eigengaps, effective rank,
   bulk shape, and seed stability.
4. Compare against OOB error, validation error, calibration, and generalization
   gap.
5. Adjust forest regularization knobs.
6. Repeat across a controlled sweep.

The forest hyperparameters most directly connected to the spectrum are:

- `max_depth`, which controls fragmentation.
- `min_samples_leaf`, which smooths leaves and reduces memorization.
- `min_samples_split`, which prevents overly local splits.
- `max_leaf_nodes`, which directly caps the feature-map dimension.
- `max_features`, which changes tree diversity and correlation.
- `n_estimators`, which stabilizes the proximity estimate.
- bootstrap sampling fraction, when available, which changes OOB coverage and
  partition diversity.

A reasonable selection principle is:

$$
\text{Choose the simplest forest with strong performance, stable alpha,}
$$

$$
\text{controlled spike structure, moderate effective rank, and small}
$$

$$
\text{full-vs-OOB spectral mismatch.}
$$

## 11. Flat Clustering Recipe

Random forest clustering often trains a forest as a discriminator between real
samples and synthetic samples, then uses terminal leaf assignments as an
embedding. This is the strategy used by the referenced
[RandomForestClustering forest embedding](https://github.com/joshloyal/RandomForestClustering/blob/master/forest_cluster/forest_embedding.py):
generate synthetic samples, train a forest to distinguish real from synthetic,
then one-hot encode `apply(X)` leaf indices.

The resulting real-sample leaf matrix $Z$ defines

$$
K = ZZ^\top / T.
$$

A concrete flat clustering recipe is:

1. Generate synthetic contrast samples from the empirical feature distribution.
2. Train a discriminative random forest to separate real from synthetic samples.
3. Extract leaf indices for the real samples.
4. Build the sparse leaf feature matrix $Z$.
5. Compute $K = ZZ^\top / T$, or use $Z$ directly.
6. Inspect the spectrum of $K$.
7. Use eigengaps and stable spectral spikes to propose candidate cluster counts.
8. Cluster with spectral clustering, kernel k-means, or k-means on a spectral
   embedding.
9. Validate cluster stability across forest seeds and synthetic resamples.

Interpretation:

- One dominant mode suggests one coarse population.
- $k$ dominant modes followed by a gap suggests $k$ candidate clusters.
- A heavy continuous tail suggests multiscale structure rather than a clean flat
  partition.
- Many unstable small spikes suggest over-partitioning or noise.

### Step 8 in detail: choosing a clustering surface

Step 8 of the recipe is a single sentence but compresses three distinct
algorithms that operate on three different objects derived from the same
forest. The right choice depends on what the downstream estimator consumes.

Let

$$
Z \in \{0, 1\}^{n \times L}, \qquad K = Z Z^\top / T, \qquad U = [u_1, \dots, u_d]
$$

denote the sparse leaf-indicator matrix, the proximity kernel, and the matrix
of leading eigenvectors of $K$. The three flat-clustering surfaces are:

1. **K-means on a spectral embedding of $K$ (recommended default).**
   Compute the top eigenvectors of $K$ and run k-means on rows of $U$
   (optionally row-normalized). This is what
   [`yggdrasil.clustering.DiscriminativeForestClusterer`](../src/yggdrasil/clustering/discriminative.py)
   ships. Prefer this surface when:
   - the spectrum has a clear eigengap;
   - cluster compactness in the kernel-induced feature space is the goal;
   - selection and label assignment should share a single eigendecomposition.

   Note that the leading eigenvector of a forest proximity kernel is usually a
   global "everything connected" mode that does not separate clusters. Drop
   it before scoring eigengaps and before forming $U$ (see section 6 on
   eigengaps and the `drop_leading_mode` parameter on
   `DiscriminativeForestClusterer`).

2. **Spectral clustering directly on $K$.**
   Use $K$ as a precomputed affinity in a normalized-cut style algorithm,
   e.g. [`sklearn.cluster.SpectralClustering`](
   https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)
   with `affinity="precomputed"`. Prefer this when:
   - $K$ has clearly disconnected blocks;
   - the user wants an off-the-shelf graph-partitioning code path;
   - they are comfortable with the implicit normalization choices that
     graph-cut packages make on their behalf.

3. **Kernel k-means on $K$.**
   Cluster samples so they are compact in the implicit feature space whose
   Gram matrix is $K$. The kernel trick avoids materializing centroids; for
   $K = Z Z^\top / T$ this is equivalent to k-means on the explicit feature
   space spanned by leaf indicators. Library options today:
   - [`tslearn.clustering.KernelKMeans`](
     https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KernelKMeans.html)
     accepts `kernel="precomputed"`. The library is otherwise a time-series
     toolkit, which is a heavier dependency than is justified for this use
     case alone.
   - A small in-house `PrecomputedKernelKMeans` that consumes either $K$ or
     directly $Z$ via the kernel trick. Yggdrasil does not currently ship one.

   Yggdrasil does **not** expose a kernel-k-means backend in v1; it is
   documented here as a recognized alternative.

A few practical cautions that apply to all three surfaces:

- **Cluster only real rows.** The synthetic rows used by the discriminator
  forest are training contrast only; the forest is applied back to the
  original real samples and the kernel/embedding is built from those rows
  only. If working manually with the doubled discriminator dataset, filter
  to the real-class indices before computing $Z$ and $K$.
- **Do not run Euclidean k-means on raw `forest.apply(X)` output.** The
  values are categorical leaf identifiers; leaf id $7$ is not "closer" to
  $8$ than to $400$. Always one-hot encode (i.e., build $Z$) before any
  Euclidean operation.
- **Choose $k$ carefully.** Eigengap proposals are noisy on small forests
  and on weakly-clustered data; pair them with spike count, modal-$k$
  voting across reseeded forests, and seed-stability checks before
  trusting a particular $k$. See sections 6 and 7.

## 12. Hierarchical Clustering Recipe

The proximity kernel can also produce hierarchical clusters.

Convert proximity to dissimilarity:

$$
D_{ij} = 1 - K_{ij}.
$$

Then run agglomerative clustering on $D$. Average linkage is a conservative
default because $D = 1-K$ is a dissimilarity but not guaranteed to be a strict
metric.

A direct hierarchical recipe is:

1. Build $K$ or $K^{OOB}$.
2. Convert to $D = 1-K$.
3. Run average-linkage or complete-linkage hierarchical clustering.
4. Use the dendrogram to identify nested structure.
5. Use eigengaps and spectral spikes to identify major hierarchy levels.
6. Use seed stability to decide which branches are reliable.

A more geometric recipe is:

1. Build $K$ or $K^{OOB}$.
2. Compute a spectral embedding from the top eigenvectors of $K$.
3. Run hierarchical clustering in that embedding.
4. Compare dendrogram stability across forest seeds and synthetic draws.

The second route is often more stable when the raw proximity matrix is noisy.

## 13. Partition Traps

The random-forest analogue of a boosting trap is a partition trap.

A partition trap is a group of samples that the forest repeatedly locks
together, or repeatedly isolates, for reasons that may not reflect robust
semantic structure.

Possible causes include:

- duplicate or near-duplicate records
- target leakage
- rare categorical artifacts
- spurious high-cardinality features
- tiny leaves created by overly deep trees
- shortcut variables that dominate the partition geometry

Spectrally, partition traps may appear as isolated spikes, localized top
eigenvectors, or spectral structure that is strong in the full kernel but weak
or unstable in the OOB kernel.

## 14. Recommended Experimental Protocol

Use controlled sweeps rather than single models.

For each dataset:

1. Train forests over a grid of complexity settings.
2. For each forest, compute full $K$, OOB $K^{OOB}$, and optionally $Z$.
3. Compute alpha, eigengaps, spike count, effective rank, and top-eigenvector
   localization.
4. Record OOB error, validation error, calibration error, and train-validation
   gap.
5. Repeat across multiple random seeds.
6. Compare spectral stability to predictive stability.
7. For clustering, compare cluster counts and hierarchies across seeds and
   synthetic resamples.

The core claim to test is:

$$
\text{Random forest regularization changes leaf-partition geometry,}
$$

$$
\text{and the OOB proximity spectrum measures that geometry out of sample.}
$$

If the claim holds, alpha and related spectral signatures become structural
feedback signals for random forest selection, clustering, and hierarchy
discovery.
