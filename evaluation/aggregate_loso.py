"""Aggregate per-fold LOSO results into one summary table.

Reads CSV files matching ``results/loso_*.csv`` (one row per fold per
model is fine, multiple rows are aggregated by model name) and writes:

* ``results/loso_summary.csv`` — model, n_folds, mean, std, sem, 95% CI
  for every numeric metric column.
* ``results/loso_per_fold.csv`` — concatenated long-format dump.

The script tolerates missing columns and ignores non-numeric ones.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable, List

import numpy as np
import pandas as pd


def load_fold_csvs(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files matched {pattern!r}. "
            "Run the LOSO benchmarks first "
            "(`bash scripts/run_loso_benchmarks.sh <MODEL>`) "
            "so that `results/loso_<model>_<masked|unmasked>.csv` files exist."
        )
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source"] = os.path.basename(p)
        if "fold" not in df.columns:
            # Derive fold id from filename, e.g. loso_seq3.csv -> 3
            base = os.path.splitext(os.path.basename(p))[0]
            df["fold"] = base.replace("loso_", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame, group_col: str = "model") -> pd.DataFrame:
    if group_col not in df.columns:
        raise KeyError(f"Expected column {group_col!r} in input CSVs.")
    numeric_cols: Iterable[str] = df.select_dtypes(include=[np.number]).columns
    summary_rows = []
    for name, sub in df.groupby(group_col):
        n = len(sub)
        row = {group_col: name, "n_folds": n}
        for col in numeric_cols:
            vals = sub[col].dropna().to_numpy()
            if vals.size == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            sem = std / np.sqrt(vals.size) if vals.size > 1 else 0.0
            ci = 1.96 * sem
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_sem"] = sem
            row[f"{col}_ci95"] = ci
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(group_col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern",
        default="results/loso_*.csv",
        help="Glob pattern of per-fold CSV files.",
    )
    ap.add_argument("--group-col", default="model")
    ap.add_argument("--out-summary", default="results/loso_summary.csv")
    ap.add_argument("--out-long", default="results/loso_per_fold.csv")
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit cleanly (instead of erroring) when no input CSVs match.",
    )
    args = ap.parse_args()

    try:
        df = load_fold_csvs(args.pattern)
    except FileNotFoundError as e:
        if args.allow_empty:
            print(f"[aggregate_loso] {e}\n[aggregate_loso] --allow-empty set, exiting cleanly.")
            return
        raise
    os.makedirs(os.path.dirname(args.out_long) or ".", exist_ok=True)
    df.to_csv(args.out_long, index=False)

    summary = summarise(df, group_col=args.group_col)
    summary.to_csv(args.out_summary, index=False)
    print(f"Wrote {args.out_long} ({len(df)} rows)")
    print(f"Wrote {args.out_summary} ({len(summary)} models)")


if __name__ == "__main__":
    main()
