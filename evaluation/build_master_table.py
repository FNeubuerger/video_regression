"""Build the master results table.

Consolidates per-model evaluation outputs into:

* ``results/MASTER_RESULTS.csv``
* ``results/MASTER_RESULTS.tex`` (booktabs LaTeX, ready for the paper)

Inputs are simple per-model CSVs in ``results/`` (one row per model with
metric columns MAE, RMSE, NLL, ECE, PICP_95, MPIW_95, params, latency_ms,
loso_gap, ...).  Missing columns are filled with ``NA`` rather than
erroring out so the table can grow incrementally.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

import pandas as pd


CANONICAL_COLUMNS = [
    "model",
    "MAE",
    "RMSE",
    "NLL",
    "ECE",
    "PICP_95",
    "MPIW_95",
    "params_M",
    "latency_ms",
    "loso_gap",
]


def load_all(patterns: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    seen: set = set()
    for pattern in patterns:
        for p in sorted(glob.glob(pattern)):
            if p in seen:
                continue
            seen.add(p)
            df = pd.read_csv(p)
            if "model" not in df.columns:
                df["model"] = os.path.splitext(os.path.basename(p))[0]
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSVs matched any of {patterns!r}")
    return pd.concat(frames, ignore_index=True, sort=False)


def to_latex_booktabs(df: pd.DataFrame) -> str:
    return df.to_latex(
        index=False,
        float_format="%.3f",
        na_rep="--",
        caption="Master results across all models and evaluation criteria.",
        label="tab:master_results",
        escape=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob pattern for input CSVs. Repeat to add more sources.",
    )
    ap.add_argument("--out-csv", default="results/MASTER_RESULTS.csv")
    ap.add_argument("--out-tex", default="results/MASTER_RESULTS.tex")
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit cleanly (instead of erroring) when no input CSVs match.",
    )
    args = ap.parse_args()

    patterns = args.pattern or [
        "results/model_*.csv",
        "results/tables/*.csv",
        "results/loso_summary.csv",
    ]
    try:
        df = load_all(patterns)
    except FileNotFoundError as e:
        if args.allow_empty:
            print(f"[build_master_table] {e}\n[build_master_table] --allow-empty set, exiting cleanly.")
            return
        raise
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[CANONICAL_COLUMNS + [c for c in df.columns if c not in CANONICAL_COLUMNS]]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    with open(args.out_tex, "w", encoding="utf-8") as fh:
        fh.write(to_latex_booktabs(df))
    print(f"Wrote {args.out_csv} and {args.out_tex} ({len(df)} rows).")


if __name__ == "__main__":
    main()
