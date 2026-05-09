# Experiment B — Partition-trap robustness

## Question

Starting from a clean 3-blob baseline, which partition-trap variants cause
v2 to misidentify `k`? For each variant, which signal "breaks" first
(picks a `k` different from the dataset's effective `k_true`), and how
much ARI does v2 lose to ground truth?

## Trap variants

We test:

- **clean** — control 3 isotropic blobs, `k_true = 3`.
- **dup_5 / dup_12 / dup_20** — clean baseline + 5/12/20 near-duplicates of a
  single row in cluster 0. Same `k_true = 3` (the duplicates belong to
  cluster 0). The trap is the rank-1 spectral mode the in-bag kernel reports
  for the duplicate group.
- **tiny_dense_5 / tiny_dense_10** — clean baseline + 5/10 rows tightly
  clustered at a non-cluster location. New label, so `k_true = 4`. The trap
  is whether v2 detects this as a real fourth cluster.
- **shortcut** — clean baseline + a binary shortcut feature that perfectly
  splits half the dataset. `k_true = 3` unchanged. The trap is the
  forest's preference for the trivial shortcut split.

## Method

For each trap variant, evaluate every signal on `k_grid = {2, 3, 4, 5, 6, 7}`
with `n_reseeds = 3` and `n_estimators = 100`, then run the production v2
clusterer and record its verdict (picked `k`, confidence, ARI to ground
truth, gating reason).

## How to run

```bash
uv run python -m experiments.B_partition_traps.run
```

Runtime budget: under 2 minutes on a laptop.

## Outputs

- `results/summary.json` — per-(dataset, k) signal table, per-signal-winning-k
  table, and v2-verdict table. Committed.
- `results/plots/signal_curves.png` — faceted line plot of signals per trap.
  Gitignored.
- `results/plots/winning_k_heatmap.png` — per-signal-winning-k heatmap.
  Gitignored.
- `results/plots/v2_ari_bars.png` — bar chart of v2 ARI-to-truth across traps,
  colored by confidence label. Gitignored.

## What "good" looks like

On the duplicates trap, the eigengap and per-seed proposal signals should
remain at `k = 3`; failures of silhouette / rotation cost / composite to
agree are evidence of small-`k` bias.

On the tiny-dense trap, signals that recover `k = 4` (matching the injected
fourth group) are detecting the new mode; signals that stay at `k = 3` are
under-clustering it.

The v2 ARI-to-truth bar chart is the single-number summary: anything below
the clean baseline's ARI on a same-`k_true` trap is a v2 failure.
