# Yggdrasil experiments

Reproducible empirical studies that don't belong in the unit-test suite —
either because they are too long-running, too dataset-heavy, or too
exploratory. Each experiment lives in its own subfolder with a `README.md`
explaining the question, a `run.py` entry point, and a `results/` directory
containing a small JSON summary (committed) and any plots (gitignored).

## Layout

```
experiments/
├── common/                       # shared dataset generators, signal evaluators, reporting
├── A_signal_characterization/    # how does each composite signal behave on clean blobs?
├── B_partition_traps/            # which signal breaks first when we inject a partition trap?
└── README.md                     # this file
```

## Running an experiment

Each experiment is a standalone module. Run from the repo root with `uv run`:

```bash
uv run python -m experiments.A_signal_characterization.run
uv run python -m experiments.B_partition_traps.run
```

Each run writes:

- `experiments/<name>/results/summary.json` — small numerical summary, committed.
- `experiments/<name>/results/plots/*.png` — seaborn plots, gitignored.

Run budgets target < 2 minutes on a laptop at default settings.

## Why these experiments

The composite cluster-count selector in
[`yggdrasil.clustering.SpectralClusterCountSelector`](../src/yggdrasil/clustering/selector.py)
combines several signals (silhouette, label stability, rotation cost,
eigengap support, modularity) into a single z-scored weighted sum. Each
signal has its own bias and variance pattern, and at v2 we suspect the
combination is fragile to partition-trap-style anomalies.

These experiments characterize each signal's behavior independently
(Theme A) and stress-test the composite against trap datasets where v2's
verdict diverges from ground truth (Theme B). They produce the evidence
needed to decide on v3 anchoring rules, OOB kernel work, or weight
retuning.
