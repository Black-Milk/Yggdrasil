# Experiment A — Signal characterization on clean blob datasets

## Question

For each diagnostic signal consumed by the v2 composite cluster-count selector
(silhouette, label stability, rotation cost, eigengap support, and ARI of
spectral-k-means against ground truth), how does the signal value vary across
candidate `k` on a sweep of *clean* synthetic-blob datasets at varying
ground-truth `k` and cluster separation?

## Method

For each `(k_true, cluster_std)` combination in the sweep, we generate an
isotropic Gaussian blob dataset with `n_per_center = 40` and 4 features, fit
the discriminative-forest embedding for `n_reseeds = 3` independent seeds, and
for every candidate `k` in `{2, ..., 7}` compute the per-(k, reseed) signal
values. The reported numbers are means and standard deviations across
reseeds.

The "winning" `k` per signal is the `k` that maximizes (or minimizes,
for rotation cost) the signal value on each dataset. Comparing the winning
`k` to `k_true` per signal tells us which signals are biased toward small
or large `k` even on clean data.

## How to run

```bash
uv run python -m experiments.A_signal_characterization.run
```

Runtime budget: under 2 minutes on a laptop at the default settings.

## Outputs

- `results/summary.json` — full per-(dataset, k) signal table plus the
  per-signal-winning-k summary. Committed.
- `results/plots/signal_curves.png` — faceted line plot of every signal vs `k`,
  one facet per dataset. Gitignored.
- `results/plots/winning_k_heatmap.png` — heatmap of the `k` each signal would
  pick on each dataset, alongside ground-truth `k`. Gitignored.

## What "good" looks like

On easy data (well-separated, modest `k`), every signal should converge on
`k_winner == k_true`. Discrepancies on this baseline are the signal's
intrinsic bias, against which we then measure trap-induced failures in
Experiment B.
