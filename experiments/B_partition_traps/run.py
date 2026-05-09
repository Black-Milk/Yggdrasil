"""Experiment B driver: partition-trap battery vs. v2 composite selector.

Run from the repo root with::

    uv run python -m experiments.B_partition_traps.run

Writes ``results/summary.json`` and three seaborn plots to
``results/plots/``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from experiments.common.datasets import (
    LabeledDataset,
    add_near_duplicates,
    add_shortcut_feature,
    add_tiny_dense_outlier_cluster,
    clean_blobs,
)
from experiments.common.reporting import (
    plot_per_signal_winning_k,
    plot_signal_curves,
    plot_v2_verdicts,
    results_dirs,
    signals_long_dataframe,
    write_summary_json,
)
from experiments.common.signals import (
    SignalRecord,
    best_k_per_signal,
    evaluate_signals,
    v2_verdict,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
K_GRID = (2, 3, 4, 5, 6, 7)
N_ESTIMATORS = 100
N_RESEEDS = 3


def build_datasets() -> list[LabeledDataset]:
    base = clean_blobs(
        n_centers=3, n_per_center=40, n_features=4, cluster_std=0.5, random_state=0, name="clean"
    )
    return [
        base,
        add_near_duplicates(base, n_duplicates=5, random_state=1, name="dup_5"),
        add_near_duplicates(base, n_duplicates=12, random_state=1, name="dup_12"),
        add_near_duplicates(base, n_duplicates=20, random_state=1, name="dup_20"),
        add_tiny_dense_outlier_cluster(base, n_extra=5, random_state=2, name="tiny_dense_5"),
        add_tiny_dense_outlier_cluster(base, n_extra=10, random_state=2, name="tiny_dense_10"),
        add_shortcut_feature(base, random_state=3, name="shortcut"),
    ]


def main() -> None:
    started = time.perf_counter()
    datasets = build_datasets()
    print(f"[B] {len(datasets)} datasets, n_reseeds={N_RESEEDS}, k_grid={K_GRID}")

    all_records: list[SignalRecord] = []
    winners_rows: list[dict] = []
    verdicts_rows: list[dict] = []
    for ds in datasets:
        t0 = time.perf_counter()
        records = evaluate_signals(
            ds.X,
            ds.y,
            dataset_name=ds.name,
            k_grid=K_GRID,
            n_estimators=N_ESTIMATORS,
            n_reseeds=N_RESEEDS,
            random_state=0,
        )
        all_records.extend(records)
        winners = best_k_per_signal(records)
        winners_rows.extend(
            {
                "dataset": ds.name,
                "signal": signal,
                "k_winner": int(k_pick),
                "k_true": int(ds.k_true),
            }
            for signal, k_pick in winners.items()
        )
        verdict = v2_verdict(
            ds.X, ds.y, dataset_name=ds.name, n_estimators=N_ESTIMATORS, random_state=0
        )
        verdicts_rows.append({"k_true": ds.k_true, **verdict.as_dict()})
        elapsed = time.perf_counter() - t0
        print(
            f"  - {ds.name:<18}  k_true={ds.k_true} "
            f"v2_picked={verdict.n_clusters_picked} "
            f"conf={verdict.confidence:<6} "
            f"ari={verdict.ari_to_truth:.3f} "
            f"({elapsed:.1f}s)"
        )

    results_dir, plots_dir = results_dirs(EXPERIMENT_DIR)
    long_df = signals_long_dataframe(all_records)
    winners_df = pd.DataFrame(winners_rows)
    verdicts_df = pd.DataFrame(verdicts_rows)

    plot_signal_curves(
        long_df,
        out_path=plots_dir / "signal_curves.png",
        title="Per-signal value across candidate k (partition traps)",
    )
    plot_per_signal_winning_k(
        winners_df,
        out_path=plots_dir / "winning_k_heatmap.png",
        title="k each signal would pick alone (partition traps)",
    )
    plot_v2_verdicts(
        verdicts_df,
        out_path=plots_dir / "v2_ari_bars.png",
        title="v2 ARI vs ground truth across partition traps",
    )

    payload = {
        "config": {
            "k_grid": list(K_GRID),
            "n_estimators": N_ESTIMATORS,
            "n_reseeds": N_RESEEDS,
            "n_datasets": len(datasets),
        },
        "per_dataset": [r.as_dict() for r in all_records],
        "per_signal_winners": winners_rows,
        "v2_verdicts": verdicts_rows,
    }
    summary_path = results_dir / "summary.json"
    write_summary_json(summary_path, payload)

    elapsed = time.perf_counter() - started
    print(f"[B] wrote {summary_path}  ({elapsed:.1f}s total)")


if __name__ == "__main__":
    main()
