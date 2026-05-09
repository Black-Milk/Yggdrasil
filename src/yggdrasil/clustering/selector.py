"""Multi-signal cluster-count selector for forest leaf kernels.

This module hosts the public ``ClusterSelectionResult`` dataclass and
the ``SpectralClusterCountSelector`` class. The selector composes the
diagnostics in :mod:`yggdrasil.clustering.diagnostics` into a final
choice of ``n_clusters``:

- ``strategy="eigengap"`` reproduces the v1 single-signal eigengap
  rule bit-identically: relative eigengaps propose ``k``, modal-k
  voting across reseeded forests confirms it, and the result falls
  back to ``k = 2`` when no clear gap is present.
- ``strategy="composite"`` (default in v2) instead gathers a small
  candidate set from each spectrum (top eigengaps + cumulative
  spectral mass), constrains it by effective rank and optional
  eigenvector-localization filters, and scores each candidate by a
  z-score-normalized weighted sum of silhouette, label stability
  across reseeds, rotation cost, and modularity. A discriminator
  out-of-bag AUC gate fires before any spectrum-shape work when the
  forest has not learned anything.

The composite scorer requires per-candidate labelings (and optionally
embeddings or kernels for the partition signals); these are passed in
via :class:`CandidateInputs`. The clusterer in
:mod:`yggdrasil.clustering.discriminative` is responsible for
producing them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from yggdrasil.clustering.diagnostics.partition import (
    modularity_on_kernel,
    silhouette_on_embedding,
)
from yggdrasil.clustering.diagnostics.rotation import rotation_cost
from yggdrasil.clustering.diagnostics.spectrum import (
    LeafSpectrum,
    cumulative_spectral_mass,
    effective_rank,
    eigengap_curve,
    inverse_participation_ratios,
)
from yggdrasil.clustering.diagnostics.stability import label_stability

__all__ = [
    "CandidateInputs",
    "ClusterSelectionResult",
    "SpectralClusterCountSelector",
]


@dataclass(frozen=True)
class CandidateInputs:
    """Per-candidate inputs consumed by the composite scorer.

    All fields are keyed by candidate ``k`` (the user-facing cluster
    count). Each value is a list with one entry per reseeded spectrum,
    matching the order of ``spectra`` passed to
    :meth:`SpectralClusterCountSelector.select`.

    Attributes
    ----------
    labelings : dict mapping int -> list of ndarray of shape (n_samples,)
        Per-(k, reseed) labelings produced by running k-means on the
        spectral embedding. Drives stability scoring.
    embeddings : dict mapping int -> list of ndarray of shape (n_samples, n_components)
        Per-(k, reseed) spectral embeddings. Drives silhouette and
        rotation-cost scoring.
    kernels : list of ndarray of shape (n_samples, n_samples) or None
        Per-reseed dense leaf kernels. Optional; required only when
        ``modularity_weight > 0`` on the selector. Materializing one
        dense kernel costs ``O(n_samples^2)`` memory.
    """

    labelings: dict[int, list[np.ndarray]] = field(default_factory=dict)
    embeddings: dict[int, list[np.ndarray]] = field(default_factory=dict)
    kernels: list[np.ndarray] | None = None


@dataclass(frozen=True)
class ClusterSelectionResult:
    """Inspectable summary of how ``n_clusters`` was chosen.

    Parameters
    ----------
    n_clusters : int
        The chosen number of clusters.
    eigenvalues : ndarray of shape (n_components,)
        Eigenvalues of the primary leaf kernel, descending.
    eigengaps : ndarray of shape (n_components - 1,)
        Relative eigengaps ``(λ_i - λ_{i+1}) / (λ_{i+1} + ε)`` over the
        full computed spectrum.
    effective_rank : float
        Spectral entropy effective rank, ``exp(H)`` with ``H`` the
        Shannon entropy of normalized eigenvalues.
    localization : ndarray of shape (n_components,)
        Inverse participation ratios of each eigenvector; small values
        indicate a broadly distributed mode, large values indicate a
        mode concentrated on few samples.
    proposed_k_per_seed : ndarray of shape (n_seeds,)
        The eigengap-proposed ``k`` from each spectrum that was passed
        to the selector.
    strategy : str
        Selection strategy used (e.g. ``"eigengap"``, ``"composite"``,
        ``"explicit"``).
    confidence : {"high", "medium", "low"}
        Self-assessment of the result. ``"low"`` is used for fallbacks
        and for AUC-gated rejections; ``"medium"`` indicates partial
        agreement of the selection signals; ``"high"`` indicates a
        clearly resolved winner.
    cumulative_mass_k : int or None
        ``k`` proposed by the cumulative-spectral-mass rule on the
        primary spectrum. ``None`` when the strategy did not consult
        cumulative mass.
    silhouette_per_k : dict mapping int -> float
        Mean silhouette across reseeds, per candidate ``k``. Empty
        when composite scoring did not run.
    stability_per_k : dict mapping int -> float
        Mean pairwise ARI across reseeds, per candidate ``k``. Empty
        when composite scoring did not run.
    rotation_cost_per_k : dict mapping int -> float
        Mean Zelnik-Manor and Perona rotation cost across reseeds, per
        candidate ``k``. Empty when composite scoring did not run.
    modularity_per_k : dict mapping int -> float
        Mean Newman modularity across reseeds, per candidate ``k``.
        Empty when modularity was not scored.
    composite_score_per_k : dict mapping int -> float
        Final z-scored weighted composite per candidate ``k``. Empty
        when composite scoring did not run.
    discriminator_auc : float or None
        Mean discriminator out-of-bag AUC across reseeds, when the
        clusterer supplied it. ``None`` otherwise.
    gating_reason : str or None
        Non-``None`` when the selector short-circuited to a fallback;
        names the rule that fired (e.g.
        ``"discriminator_auc_below_threshold"``,
        ``"empty_candidate_set"``, ``"no_structure"``).
    """

    n_clusters: int
    eigenvalues: np.ndarray
    eigengaps: np.ndarray
    effective_rank: float
    localization: np.ndarray
    proposed_k_per_seed: np.ndarray
    strategy: str
    confidence: Literal["high", "medium", "low"]
    cumulative_mass_k: int | None = None
    silhouette_per_k: dict[int, float] = field(default_factory=dict)
    stability_per_k: dict[int, float] = field(default_factory=dict)
    rotation_cost_per_k: dict[int, float] = field(default_factory=dict)
    modularity_per_k: dict[int, float] = field(default_factory=dict)
    composite_score_per_k: dict[int, float] = field(default_factory=dict)
    discriminator_auc: float | None = None
    gating_reason: str | None = None


def _safe_finite_mean(arr: np.ndarray) -> float:
    """Mean of an array's finite entries, or ``0.0`` if there are none."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def _fill_nans(arr: np.ndarray) -> np.ndarray:
    """Replace ``nan``/``inf`` entries with the array's finite-mean."""
    out = np.asarray(arr, dtype=np.float64).copy()
    mask = ~np.isfinite(out)
    if not mask.any():
        return out
    out[mask] = _safe_finite_mean(out)
    return out


def _zscore(values: np.ndarray) -> np.ndarray:
    """Z-score normalize a 1-D array; return zeros if std is zero or undefined."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    mu = float(np.mean(finite))
    sd = float(np.std(finite))
    if not np.isfinite(sd) or sd <= 0.0:
        return np.zeros_like(arr)
    z = (arr - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z


class SpectralClusterCountSelector:
    """Multi-signal selector for ``n_clusters`` from leaf-kernel spectra.

    Parameters
    ----------
    max_clusters : int, default=20
        Largest cluster count considered.
    drop_leading_mode : bool, default=True
        If ``True``, the first (largest) eigenvalue is excluded when
        proposing ``k`` from eigengaps. The leading eigenvector of a
        proximity kernel is usually a global "everything connected"
        mode and dominates eigengaps without reflecting cluster
        structure.
    min_relative_gap : float, default=0.5
        Minimum relative eigengap required for a spectrum to count as
        "structured" in eigengap-mode confidence scoring. Inactive in
        composite mode.
    strategy : {"eigengap", "composite"}, default="composite"
        Selection rule. ``"eigengap"`` reproduces the v1 behavior
        bit-identically. ``"composite"`` combines eigengap, cumulative
        spectral mass, silhouette, label stability, rotation cost,
        and modularity into a single z-scored weighted sum.
    min_discriminator_auc : float, default=0.55
        Composite-mode AUC gate. When the supplied
        ``discriminator_auc`` is below this threshold the selector
        returns ``k = 2`` with ``confidence = "low"`` and
        ``gating_reason = "discriminator_auc_below_threshold"``.
    cumulative_mass_threshold : float, default=0.9
        Threshold passed to
        :func:`yggdrasil.clustering.diagnostics.cumulative_spectral_mass`
        when proposing a candidate ``k`` from cumulative spectral mass.
    silhouette_weight : float, default=1.0
        Weight applied to the z-scored silhouette signal in composite
        scoring. Higher silhouette is better.
    stability_weight : float, default=1.0
        Weight applied to the z-scored label-stability signal in
        composite scoring. Higher stability is better.
    rotation_weight : float, default=1.0
        Weight applied to the negated z-scored rotation-cost signal
        in composite scoring. Lower rotation cost is better, so it is
        negated before being added.
    modularity_weight : float, default=0.0
        Weight applied to the z-scored Newman-modularity signal in
        composite scoring. Off by default because materializing a
        dense leaf kernel for modularity is ``O(n_samples^2)``.
    localization_threshold : float or None, default=None
        If set, candidate ``k`` is rejected when any of the leading
        ``k`` eigenvectors of the primary spectrum has an inverse
        participation ratio above this threshold (i.e. is overly
        concentrated on a few samples). Use small positive values like
        ``0.5`` for adversarial filtering.
    confidence_margin : float, default=0.5
        Minimum gap (in z-score units) between the winning composite
        score and the runner-up required for ``confidence = "high"``.
        Otherwise the selector reports ``"medium"`` (when stability
        clears the floor) or ``"low"``.
    stability_floor : float, default=0.6
        Minimum mean ARI across reseeds at the winning ``k`` required
        for ``confidence = "high"``. With fewer reseeds, stability is
        ``nan`` and treated as not failing the floor.

    Notes
    -----
    The selector never raises on noisy or degenerate spectra; it
    always returns ``k >= 2``. This is what makes it compatible with
    sklearn ``check_estimator`` runs that exercise tiny synthetic
    datasets where every signal is essentially noise.
    """

    def __init__(
        self,
        *,
        max_clusters: int = 20,
        drop_leading_mode: bool = True,
        min_relative_gap: float = 0.5,
        strategy: Literal["eigengap", "composite"] = "composite",
        min_discriminator_auc: float = 0.55,
        cumulative_mass_threshold: float = 0.9,
        silhouette_weight: float = 1.0,
        stability_weight: float = 1.0,
        rotation_weight: float = 1.0,
        modularity_weight: float = 0.0,
        localization_threshold: float | None = None,
        confidence_margin: float = 0.5,
        stability_floor: float = 0.6,
    ) -> None:
        self.max_clusters = max_clusters
        self.drop_leading_mode = drop_leading_mode
        self.min_relative_gap = min_relative_gap
        self.strategy = strategy
        self.min_discriminator_auc = min_discriminator_auc
        self.cumulative_mass_threshold = cumulative_mass_threshold
        self.silhouette_weight = silhouette_weight
        self.stability_weight = stability_weight
        self.rotation_weight = rotation_weight
        self.modularity_weight = modularity_weight
        self.localization_threshold = localization_threshold
        self.confidence_margin = confidence_margin
        self.stability_floor = stability_floor

    def candidate_set(self, spectra: Sequence[LeafSpectrum]) -> list[int]:
        """Return composite-mode candidate ``k`` values from a list of spectra.

        For each spectrum the selector takes the top three eigengap
        candidates plus the cumulative-spectral-mass candidate, then
        unions across reseeds and intersects with
        ``[2, min(max_clusters, ceil(mean_effective_rank))]``. The
        optional ``localization_threshold`` filter is applied
        afterwards using the primary spectrum's eigenvector
        localization.

        Parameters
        ----------
        spectra : sequence of LeafSpectrum
            One spectrum per reseeded forest.

        Returns
        -------
        candidates : list of int
            Sorted, deduplicated list of candidate ``k`` values. The
            list may be empty when no spectrum supports a ``k >= 2``.
        """
        if not spectra:
            return []

        per_spectrum: set[int] = set()
        eff_ranks: list[float] = []
        for spectrum in spectra:
            eigs = np.asarray(spectrum.eigenvalues, dtype=np.float64)
            eff_ranks.append(effective_rank(eigs))
            per_spectrum.update(self._eigengap_candidates(eigs, top=3))
            per_spectrum.add(
                cumulative_spectral_mass(spectrum, threshold=self.cumulative_mass_threshold)
            )
        max_eff_rank = max(2, int(np.ceil(float(np.mean(eff_ranks)))))
        upper = max(2, int(min(self.max_clusters, max_eff_rank)))

        filtered = sorted(k for k in per_spectrum if 2 <= k <= upper)

        if self.localization_threshold is not None and filtered:
            primary = spectra[0]
            ipr = inverse_participation_ratios(primary.eigenvectors)
            n_components = ipr.shape[0]
            offset = 1 if (self.drop_leading_mode and n_components > 1) else 0

            def _ok(k: int) -> bool:
                end = min(offset + k, n_components)
                if end <= offset:
                    return True
                return bool(np.max(ipr[offset:end]) <= self.localization_threshold)

            filtered = [k for k in filtered if _ok(k)]
        return filtered

    def select(
        self,
        spectra: Sequence[LeafSpectrum],
        *,
        candidate_inputs: CandidateInputs | None = None,
        discriminator_auc: float | None = None,
    ) -> ClusterSelectionResult:
        """Select ``n_clusters`` from one or more leaf-kernel spectra.

        Parameters
        ----------
        spectra : sequence of LeafSpectrum
            One spectrum per re-fitted forest seed. The first element
            is treated as the "primary" spectrum for diagnostic
            fields of the returned result.
        candidate_inputs : CandidateInputs, optional
            Per-(candidate-k, reseed) labelings and embeddings used by
            composite scoring. Required for ``strategy="composite"``;
            ignored otherwise.
        discriminator_auc : float, optional
            Mean discriminator out-of-bag AUC across reseeds. Used for
            the composite-mode AUC gate; recorded on the result.

        Returns
        -------
        result : ClusterSelectionResult
            Selected ``n_clusters`` plus the supporting diagnostics.

        Raises
        ------
        ValueError
            If ``spectra`` is empty.
        """
        if not spectra:
            raise ValueError("spectra must contain at least one LeafSpectrum.")

        if self.strategy == "eigengap":
            return self._eigengap_select(spectra)
        return self._composite_select(
            spectra,
            candidate_inputs=candidate_inputs,
            discriminator_auc=discriminator_auc,
        )

    def _eigengap_select(self, spectra: Sequence[LeafSpectrum]) -> ClusterSelectionResult:
        """V1-compatible single-signal eigengap selection."""
        proposals = [self._propose_from_one(s.eigenvalues) for s in spectra]
        ks = np.array([p[0] for p in proposals], dtype=np.int64)
        gaps = np.array([p[1] for p in proposals], dtype=np.float64)
        structured = gaps >= self.min_relative_gap

        n_proposals = len(proposals)
        confidence: Literal["high", "medium", "low"] = "low"
        chosen_k = 2
        if bool(structured.any()):
            unique_vals, counts = np.unique(ks, return_counts=True)
            order = np.lexsort((unique_vals, -counts))
            modal_k = int(unique_vals[order[0]])
            modal_count = int(counts[order[0]])
            n_structured = int(structured.sum())
            has_modal_majority = 2 * modal_count > n_proposals
            has_structure_majority = 2 * n_structured > n_proposals
            if has_modal_majority and has_structure_majority:
                chosen_k = modal_k
                confidence = "high"

        primary = spectra[0]
        return ClusterSelectionResult(
            n_clusters=chosen_k,
            eigenvalues=primary.eigenvalues.copy(),
            eigengaps=eigengap_curve(primary.eigenvalues),
            effective_rank=effective_rank(primary.eigenvalues),
            localization=inverse_participation_ratios(primary.eigenvectors),
            proposed_k_per_seed=ks,
            strategy="eigengap",
            confidence=confidence,
            gating_reason=None if confidence == "high" else "no_structure",
        )

    def _composite_select(
        self,
        spectra: Sequence[LeafSpectrum],
        *,
        candidate_inputs: CandidateInputs | None,
        discriminator_auc: float | None,
    ) -> ClusterSelectionResult:
        """Multi-signal composite selection."""
        primary = spectra[0]
        primary_eigs = primary.eigenvalues
        eigengaps_full = eigengap_curve(primary_eigs)
        eff_rank = effective_rank(primary_eigs)
        ipr = inverse_participation_ratios(primary.eigenvectors)
        cum_mass_k = cumulative_spectral_mass(primary, threshold=self.cumulative_mass_threshold)
        proposed_per_seed = np.array(
            [self._propose_from_one(s.eigenvalues)[0] for s in spectra],
            dtype=np.int64,
        )

        def _result(
            n_clusters: int,
            confidence: Literal["high", "medium", "low"],
            gating_reason: str | None,
            *,
            silhouette_per_k: dict[int, float] | None = None,
            stability_per_k: dict[int, float] | None = None,
            rotation_cost_per_k: dict[int, float] | None = None,
            modularity_per_k: dict[int, float] | None = None,
            composite_score_per_k: dict[int, float] | None = None,
        ) -> ClusterSelectionResult:
            return ClusterSelectionResult(
                n_clusters=n_clusters,
                eigenvalues=primary_eigs.copy(),
                eigengaps=eigengaps_full,
                effective_rank=eff_rank,
                localization=ipr,
                proposed_k_per_seed=proposed_per_seed,
                strategy="composite",
                confidence=confidence,
                cumulative_mass_k=int(cum_mass_k),
                silhouette_per_k=silhouette_per_k or {},
                stability_per_k=stability_per_k or {},
                rotation_cost_per_k=rotation_cost_per_k or {},
                modularity_per_k=modularity_per_k or {},
                composite_score_per_k=composite_score_per_k or {},
                discriminator_auc=discriminator_auc,
                gating_reason=gating_reason,
            )

        if (
            discriminator_auc is not None
            and np.isfinite(discriminator_auc)
            and discriminator_auc < self.min_discriminator_auc
        ):
            return _result(2, "low", "discriminator_auc_below_threshold")

        if candidate_inputs is None:
            raise ValueError(
                "SpectralClusterCountSelector(strategy='composite').select(...) "
                "requires `candidate_inputs` (per-(k, reseed) labelings and "
                "embeddings). Either provide them or set strategy='eigengap'."
            )

        candidates = self.candidate_set(spectra)
        if not candidates:
            return _result(2, "low", "empty_candidate_set")

        scores: dict[int, dict[str, float]] = {}
        for k in candidates:
            silhouettes: list[float] = []
            for emb, lbl in zip(
                candidate_inputs.embeddings.get(k, []),
                candidate_inputs.labelings.get(k, []),
                strict=False,
            ):
                silhouettes.append(silhouette_on_embedding(emb, lbl))
            silhouette_mean = float(np.nanmean(silhouettes)) if silhouettes else float("nan")

            labelings = candidate_inputs.labelings.get(k, [])
            stability = (
                float(label_stability(labelings, metric="ari"))
                if len(labelings) >= 2
                else float("nan")
            )

            rotations: list[float] = []
            for emb in candidate_inputs.embeddings.get(k, []):
                top = np.asarray(emb, dtype=np.float64)
                if top.shape[1] >= 2:
                    rotations.append(rotation_cost(top, random_state=0))
            rotation_mean = float(np.nanmean(rotations)) if rotations else float("nan")

            modularities: list[float] = []
            if (
                self.modularity_weight > 0.0
                and candidate_inputs.kernels is not None
                and len(candidate_inputs.kernels) > 0
            ):
                for K, lbl in zip(
                    candidate_inputs.kernels,
                    candidate_inputs.labelings.get(k, []),
                    strict=False,
                ):
                    modularities.append(modularity_on_kernel(K, lbl))
            modularity_mean = float(np.nanmean(modularities)) if modularities else float("nan")

            scores[k] = {
                "silhouette": silhouette_mean,
                "stability": stability,
                "rotation": rotation_mean,
                "modularity": modularity_mean,
            }

        ks_arr = np.array(candidates, dtype=np.int64)
        sil_arr = np.array([scores[k]["silhouette"] for k in candidates])
        stab_arr = np.array([scores[k]["stability"] for k in candidates])
        rot_arr = np.array([scores[k]["rotation"] for k in candidates])
        mod_arr = np.array([scores[k]["modularity"] for k in candidates])

        z_sil = _zscore(_fill_nans(sil_arr))
        z_stab = _zscore(_fill_nans(stab_arr))
        z_rot = _zscore(_fill_nans(rot_arr))
        z_mod = _zscore(_fill_nans(mod_arr))

        composite = (
            self.silhouette_weight * np.nan_to_num(z_sil)
            + self.stability_weight * np.nan_to_num(z_stab)
            - self.rotation_weight * np.nan_to_num(z_rot)
            + self.modularity_weight * np.nan_to_num(z_mod)
        )

        silhouette_per_k = {int(k): float(v) for k, v in zip(ks_arr, sil_arr, strict=True)}
        stability_per_k = {int(k): float(v) for k, v in zip(ks_arr, stab_arr, strict=True)}
        rotation_cost_per_k = {int(k): float(v) for k, v in zip(ks_arr, rot_arr, strict=True)}
        modularity_per_k = {int(k): float(v) for k, v in zip(ks_arr, mod_arr, strict=True)}
        composite_score_per_k = {int(k): float(v) for k, v in zip(ks_arr, composite, strict=True)}

        winner_idx = int(np.argmax(composite))
        winner_k = int(ks_arr[winner_idx])
        winner_score = float(composite[winner_idx])
        runner_score = float(np.partition(composite, -2)[-2]) if composite.size >= 2 else -np.inf
        margin = winner_score - runner_score
        winner_stability = stab_arr[winner_idx]
        stability_ok = np.isnan(winner_stability) or float(winner_stability) >= self.stability_floor

        if margin >= self.confidence_margin and stability_ok:
            confidence: Literal["high", "medium", "low"] = "high"
            gating_reason = None
        elif stability_ok:
            confidence = "medium"
            gating_reason = "narrow_composite_margin"
        else:
            confidence = "low"
            gating_reason = "low_label_stability"
            winner_k = 2

        return _result(
            winner_k,
            confidence,
            gating_reason,
            silhouette_per_k=silhouette_per_k,
            stability_per_k=stability_per_k,
            rotation_cost_per_k=rotation_cost_per_k,
            modularity_per_k=modularity_per_k,
            composite_score_per_k=composite_score_per_k,
        )

    def _eigengap_candidates(self, eigenvalues: np.ndarray, top: int) -> list[int]:
        """Top ``top`` eigengap-proposed ``k`` values from a single spectrum.

        Mirrors :meth:`_propose_from_one`'s reasoning, but returns the
        top ``top`` candidates rather than only the largest.
        """
        eigs = np.asarray(eigenvalues, dtype=np.float64)
        offset = 1 if (self.drop_leading_mode and eigs.size > 1) else 0
        eigs_eff = eigs[offset:]
        if eigs_eff.size < 2:
            return [2]

        min_k_eff = max(1, 2 - offset)
        max_k_eff = int(min(self.max_clusters - offset, eigs_eff.size - 1))
        if max_k_eff < min_k_eff:
            return [2]

        candidate_k_eff = np.arange(min_k_eff, max_k_eff + 1)
        num = eigs_eff[candidate_k_eff - 1] - eigs_eff[candidate_k_eff]
        denom = eigs_eff[candidate_k_eff] + 1e-12
        gaps = num / denom
        order = np.argsort(gaps)[::-1][: max(1, top)]
        return [int(candidate_k_eff[i]) + offset for i in order]

    def _propose_from_one(self, eigenvalues: np.ndarray) -> tuple[int, float]:
        """Propose ``k`` and its supporting relative eigengap from one spectrum.

        With ``drop_leading_mode=True`` the largest eigenvalue is
        treated as a global "everything connected" mode and skipped.
        The remaining eigenvalues are scored at offsets ``k_eff =
        1, 2, ...`` (gap between ``λ_{k_eff}`` and ``λ_{k_eff+1}`` of
        the trimmed spectrum). The reported cluster count adds the
        dropped mode back, so ``k_eff = 1`` corresponds to
        ``n_clusters = 2``.
        """
        eigs = np.asarray(eigenvalues, dtype=np.float64)
        offset = 1 if (self.drop_leading_mode and eigs.size > 1) else 0
        eigs_eff = eigs[offset:]
        if eigs_eff.size < 2:
            return (2, 0.0)

        min_k_eff = max(1, 2 - offset)
        max_k_eff = int(min(self.max_clusters - offset, eigs_eff.size - 1))
        if max_k_eff < min_k_eff:
            return (2, 0.0)

        candidate_k_eff = np.arange(min_k_eff, max_k_eff + 1)
        num = eigs_eff[candidate_k_eff - 1] - eigs_eff[candidate_k_eff]
        denom = eigs_eff[candidate_k_eff] + 1e-12
        gaps = num / denom

        idx = int(np.argmax(gaps))
        proposed_k = int(candidate_k_eff[idx]) + offset
        return (proposed_k, float(gaps[idx]))
