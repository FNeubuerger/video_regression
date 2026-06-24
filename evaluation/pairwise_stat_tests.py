"""Pairwise statistical tests across models.

Reads ``results/loso_per_fold.csv`` (produced by ``aggregate_loso.py``)
and computes pairwise Wilcoxon signed-rank p-values on a chosen metric
(default MAE).  Output is a square ``models x models`` matrix CSV plus a
heatmap PNG.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def pairwise_wilcoxon(df: pd.DataFrame, metric: str, group_col: str = "model") -> pd.DataFrame:
    models = sorted(df[group_col].unique())
    pivot = df.pivot_table(
        index="fold", columns=group_col, values=metric, aggfunc="mean"
    ).dropna(how="any")
    n = len(models)
    pvals = np.full((n, n), np.nan)
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                pvals[i, j] = 1.0
                continue
            if m1 not in pivot.columns or m2 not in pivot.columns:
                continue
            try:
                stat = wilcoxon(pivot[m1], pivot[m2])
                pvals[i, j] = stat.pvalue
            except ValueError:
                pvals[i, j] = np.nan
    return pd.DataFrame(pvals, index=models, columns=models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/loso_per_fold.csv")
    ap.add_argument("--metric", default="MAE")
    ap.add_argument("--out-csv", default="results/pairwise_wilcoxon.csv")
    ap.add_argument("--out-png", default="results/pairwise_wilcoxon.png")
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit cleanly (instead of erroring) when the input CSV is missing.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.input):
        msg = (
            f"Input {args.input!r} not found. "
            "Run `make loso_aggregate` first (after producing some loso_*.csv folds)."
        )
        if args.allow_empty:
            print(f"[pairwise_stat_tests] {msg}\n[pairwise_stat_tests] --allow-empty set, exiting cleanly.")
            return
        raise FileNotFoundError(msg)

    df = pd.read_csv(args.input)
    p = pairwise_wilcoxon(df, args.metric)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    p.to_csv(args.out_csv)

    fig, ax = plt.subplots(figsize=(0.8 * len(p) + 2, 0.8 * len(p) + 2))
    im = ax.imshow(-np.log10(p.values + 1e-12), cmap="viridis", vmin=0, vmax=4)
    ax.set_xticks(range(len(p)))
    ax.set_yticks(range(len(p)))
    ax.set_xticklabels(p.columns, rotation=45, ha="right")
    ax.set_yticklabels(p.index)
    ax.set_title(f"-log10 pairwise Wilcoxon p-value ({args.metric})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out_csv} and {args.out_png}")


if __name__ == "__main__":
    main()
