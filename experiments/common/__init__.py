"""Shared building blocks for Yggdrasil experiments.

The submodules are kept small and orthogonal so individual experiment
scripts can import only what they need:

- :mod:`experiments.common.datasets` — synthetic dataset generators
  (clean blobs, partition traps, imbalanced clusters).
- :mod:`experiments.common.signals` — per-(k, reseed) evaluators of the
  diagnostic signals consumed by the composite selector.
- :mod:`experiments.common.reporting` — JSON summary writer and seaborn
  plotting helpers.
"""
