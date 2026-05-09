"""JSON summary writer and seaborn plotting helpers for experiments.

Conventions followed by every experiment in this repo:

- Numerical summaries (small, human-inspectable JSON) go in
  ``experiments/<exp>/results/summary.json`` and are committed.
- Plots go in ``experiments/<exp>/results/plots/<name>.png`` and are
  gitignored (see ``experiments/.gitignore``).
- Both directories are created on demand by :func:`results_dirs`.

The plotting helpers favor seaborn defaults (whitegrid, "deep" palette,
``font_scale=1.0``) and write PNGs at 150 DPI. Plots return the
``matplotlib.figure.Figure`` object so callers can extend them if
needed; the helpers themselves close the figure after writing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend for batch runs

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.0)


def results_dirs(experiment_dir: Path) -> tuple[Path, Path]:
    """Return ``(results_dir, plots_dir)`` for an experiment, creating both."""
    results = Path(experiment_dir) / "results"
    plots = results / "plots"
    results.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return results, plots


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    """Serialize a summary payload as pretty-printed JSON.

    NumPy scalars are converted to Python builtins via ``default=_to_builtin``
    so the file is round-trippable without numpy.
    """
    Path(path).write_text(json.dumps(payload, indent=2, default=_to_builtin))


def _to_builtin(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON-serializable")


def signals_long_dataframe(records: list[Any]) -> pd.DataFrame:
    """Convert a list of ``SignalRecord`` (or dicts) to a long-form dataframe.

    The output has one row per ``(dataset, k, signal)``, suitable for
    seaborn faceted plots.
    """
    rows: list[dict[str, Any]] = []
    for r in records:
        d = r.as_dict() if hasattr(r, "as_dict") else dict(r)
        for signal in (
            "silhouette_mean",
            "rotation_cost_mean",
            "label_stability",
            "eigengap_support",
            "ari_kmeans_to_truth_mean",
        ):
            rows.append(
                {
                    "dataset": d["dataset"],
                    "k_true": d["k_true"],
                    "n_samples": d["n_samples"],
                    "k": d["k"],
                    "signal": signal.replace("_mean", "").replace("_", " "),
                    "value": d[signal],
                }
            )
    return pd.DataFrame(rows)


def plot_signal_curves(
    df: pd.DataFrame,
    *,
    out_path: Path,
    title: str | None = None,
    col_wrap: int = 3,
):
    """Faceted line plot of every signal vs ``k``, one facet per dataset."""
    g = sns.relplot(
        data=df,
        x="k",
        y="value",
        hue="signal",
        kind="line",
        marker="o",
        col="dataset",
        col_wrap=col_wrap,
        height=2.6,
        aspect=1.4,
        facet_kws={"sharey": False, "sharex": True},
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("k", "value")
    if title is not None:
        g.figure.suptitle(title, y=1.02)
    g.figure.tight_layout()
    g.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)
    return g


def plot_per_signal_winning_k(
    winners_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str | None = None,
):
    """Heatmap of "k chosen by signal X on dataset Y" with the true k as a baseline.

    ``winners_df`` is expected to have columns
    ``["dataset", "signal", "k_winner", "k_true"]``.
    """
    heat = winners_df.pivot(index="dataset", columns="signal", values="k_winner")
    truth = winners_df.drop_duplicates(subset=["dataset"]).set_index("dataset")["k_true"]
    heat["__k_true__"] = truth
    heat = heat[["__k_true__", *[c for c in heat.columns if c != "__k_true__"]]]

    fig, ax = plt.subplots(figsize=(0.9 * heat.shape[1] + 2, 0.45 * heat.shape[0] + 2))
    sns.heatmap(
        heat,
        annot=True,
        fmt="d",
        cmap="viridis",
        cbar_kws={"label": "k"},
        linewidths=0.4,
        ax=ax,
    )
    ax.set_xlabel("signal (first column = ground-truth k)")
    ax.set_ylabel("")
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_v2_verdicts(
    verdicts_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str | None = None,
):
    """Bar plot of v2 ARI-to-truth across datasets, colored by confidence label."""
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(verdicts_df) + 2), 4))
    sns.barplot(
        data=verdicts_df,
        x="dataset",
        y="ari_to_truth",
        hue="confidence",
        dodge=False,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ARI(v2_labels, ground_truth)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
