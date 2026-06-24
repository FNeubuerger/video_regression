"""Calibration diagnostics for uncertainty-aware regressors.

Reads a per-sample CSV with columns ``target``, ``mean``, ``std`` and
emits:

* Reliability diagram (PNG): empirical vs nominal coverage of central
  prediction intervals.
* Sharpness (PNG): histogram of predictive std.
* Calibration metrics (CSV): ECE, MCE, sharpness mean/median, PICP, MPIW.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def empirical_coverage(targets, means, stds, alpha):
    z = norm.ppf(1 - (1 - alpha) / 2)
    lo, hi = means - z * stds, means + z * stds
    return float(((targets >= lo) & (targets <= hi)).mean())


def compute_calibration(targets, means, stds, n_bins: int = 10):
    nominal = np.linspace(0.05, 0.95, n_bins)
    empirical = np.array([empirical_coverage(targets, means, stds, a) for a in nominal])
    ece = float(np.mean(np.abs(empirical - nominal)))
    mce = float(np.max(np.abs(empirical - nominal)))
    picp = empirical_coverage(targets, means, stds, 0.95)
    mpiw = float(2 * 1.96 * stds.mean())
    return {
        "nominal": nominal,
        "empirical": empirical,
        "ECE": ece,
        "MCE": mce,
        "PICP_95": picp,
        "MPIW_95": mpiw,
        "sharpness_mean": float(stds.mean()),
        "sharpness_median": float(np.median(stds)),
    }


def plot_reliability(metrics: dict, out_path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="ideal")
    ax.plot(metrics["nominal"], metrics["empirical"], "o-", label="empirical")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title or "Reliability diagram")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sharpness(stds: np.ndarray, out_path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(stds, bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("Predictive std")
    ax.set_ylabel("Count")
    ax.set_title(title or "Sharpness")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with columns target, mean, std")
    ap.add_argument("--out-dir", default="results/calibration")
    ap.add_argument("--name", default="model")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    targets = df["target"].to_numpy()
    means = df["mean"].to_numpy()
    stds = df["std"].to_numpy()

    os.makedirs(args.out_dir, exist_ok=True)
    m = compute_calibration(targets, means, stds)
    plot_reliability(m, os.path.join(args.out_dir, f"{args.name}_reliability.png"), title=args.name)
    plot_sharpness(stds, os.path.join(args.out_dir, f"{args.name}_sharpness.png"), title=args.name)

    out_csv = os.path.join(args.out_dir, f"{args.name}_calibration.csv")
    pd.DataFrame([{
        "model": args.name,
        "ECE": m["ECE"],
        "MCE": m["MCE"],
        "PICP_95": m["PICP_95"],
        "MPIW_95": m["MPIW_95"],
        "sharpness_mean": m["sharpness_mean"],
        "sharpness_median": m["sharpness_median"],
    }]).to_csv(out_csv, index=False)
    print(f"Wrote calibration outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
