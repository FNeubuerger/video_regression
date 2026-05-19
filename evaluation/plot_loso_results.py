"""Generate LOSO-cross-validation figures and a LaTeX summary table.

Consumes:
  - ``results/loso_per_fold.csv`` (long format, one row per (model, fold))
  - ``results/loso_summary.csv``  (one row per model, with mean/std/sem/CI95)

Produces:
  - ``paper/figures/loso_mae_bars.png``       MAE +/- 95% CI bar chart
  - ``paper/figures/loso_field_mae_bars.png`` Field MAE bar chart (spatial models)
  - ``paper/figures/loso_per_fold_heatmap.png`` Per-fold MAE heatmap
  - ``paper/figures/kan_comparison.png``      KAN variants vs their baselines
  - ``paper/tables/loso_summary.tex``         Paper-ready LaTeX table

Run from the repository root:
    .venv/bin/python evaluation/plot_loso_results.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Models that produce a (B, T, h, w) field and therefore have field_mae / field_rmse.
SPATIAL_MODELS = {
    "ConvLTC",
    "ConvectionBioheat",
    "SpatialResNet",
    "SpatialKANBioheat",
}

# Pairs of (baseline, KAN-variant) for the focused KAN comparison plot.
KAN_PAIRS = [
    ("SimpleResNet", "KANResNet"),
    ("ConvectionBioheat", "SpatialKANBioheat"),
]


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["legend.fontsize"] = 9


def plot_mae_bars(summary: pd.DataFrame, out_path: str, metric: str = "mae") -> None:
    col_mean = f"{metric}_mean"
    col_ci = f"{metric}_ci95"
    df = summary.dropna(subset=[col_mean]).copy()
    df = df.sort_values(col_mean)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#d62728" if "KAN" in m else "#1f77b4" for m in df["model"]]
    ax.bar(
        df["model"],
        df[col_mean],
        yerr=df[col_ci],
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel(f"{metric.upper()} (K)  — mean $\\pm$ 95% CI across 15 LOSO folds")
    ax.set_title(f"Leave-One-Subject-Out {metric.upper()} per model")
    ax.tick_params(axis="x", labelrotation=45)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_field_mae_bars(summary: pd.DataFrame, out_path: str) -> None:
    df = summary.dropna(subset=["field_mae_mean"]).copy()
    if df.empty:
        print("No spatial models with field_mae — skipping field MAE plot.")
        return
    df = df.sort_values("field_mae_mean")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#d62728" if "KAN" in m else "#2ca02c" for m in df["model"]]
    ax.bar(
        df["model"],
        df["field_mae_mean"],
        yerr=df["field_mae_ci95"],
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Field MAE (K) — mean $\\pm$ 95% CI")
    ax.set_title("Spatial field reconstruction error (LOSO)")
    ax.tick_params(axis="x", labelrotation=20)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_per_fold_heatmap(per_fold: pd.DataFrame, out_path: str) -> None:
    pivot = per_fold.pivot_table(
        index="model", columns="fold", values="mae", aggfunc="mean"
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(11, 0.45 * len(pivot) + 2))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        cbar_kws={"label": "MAE (K)"},
        ax=ax,
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_title("Per-fold MAE across LOSO splits")
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_kan_comparison(summary: pd.DataFrame, out_path: str) -> None:
    rows = []
    for baseline, kan in KAN_PAIRS:
        for variant in (baseline, kan):
            sub = summary[summary["model"] == variant]
            if sub.empty:
                continue
            rec = sub.iloc[0]
            rows.append(
                {
                    "Pair": f"{baseline} vs {kan}",
                    "Variant": "KAN" if "KAN" in variant else "Baseline",
                    "Model": variant,
                    "MAE": rec.get("mae_mean", np.nan),
                    "MAE_CI": rec.get("mae_ci95", np.nan),
                    "FieldMAE": rec.get("field_mae_mean", np.nan),
                    "FieldMAE_CI": rec.get("field_mae_ci95", np.nan),
                }
            )
    if not rows:
        print("No KAN variants available yet — skipping KAN comparison plot.")
        return
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    palette = {"Baseline": "#1f77b4", "KAN": "#d62728"}

    sns.barplot(
        data=df, x="Pair", y="MAE", hue="Variant", ax=axes[0], palette=palette, errorbar=None
    )
    for bar, (_, row) in zip(axes[0].patches, df.iterrows()):
        if np.isfinite(row["MAE_CI"]):
            axes[0].errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                yerr=row["MAE_CI"],
                color="black",
                capsize=3,
                fmt="none",
            )
    axes[0].set_title("Scalar MAE (K)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("MAE (K)")

    df_field = df.dropna(subset=["FieldMAE"])
    if not df_field.empty:
        sns.barplot(
            data=df_field,
            x="Pair",
            y="FieldMAE",
            hue="Variant",
            ax=axes[1],
            palette=palette,
            errorbar=None,
        )
        for bar, (_, row) in zip(axes[1].patches, df_field.iterrows()):
            if np.isfinite(row["FieldMAE_CI"]):
                axes[1].errorbar(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    yerr=row["FieldMAE_CI"],
                    color="black",
                    capsize=3,
                    fmt="none",
                )
        axes[1].set_title("Spatial Field MAE (K)")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Field MAE (K)")
    else:
        axes[1].set_axis_off()

    fig.suptitle("KAN variants vs their baselines (LOSO mean $\\pm$ 95% CI)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def write_latex_table(summary: pd.DataFrame, out_path: str) -> None:
    df = summary.copy().sort_values("mae_mean")

    def fmt(mean, ci):
        if pd.isna(mean):
            return "--"
        if pd.isna(ci):
            return f"{mean:.2f}"
        return f"{mean:.2f} $\\pm$ {ci:.2f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Leave-one-subject-out cross-validation results (15 folds). "
        r"Values are mean $\pm$ 95\% confidence interval across folds. "
        r"Field MAE/RMSE are reported only for models that produce a spatial output.}",
        r"\label{tab:loso_results}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{MAE} (K) $\downarrow$ & \textbf{RMSE} (K) $\downarrow$ "
        r"& \textbf{Field MAE} (K) $\downarrow$ & \textbf{Field RMSE} (K) $\downarrow$ \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        name = row["model"].replace("_", r"\_")
        mae = fmt(row.get("mae_mean"), row.get("mae_ci95"))
        rmse = fmt(row.get("rmse_mean"), row.get("rmse_ci95"))
        fmae = fmt(row.get("field_mae_mean"), row.get("field_mae_ci95"))
        frmse = fmt(row.get("field_rmse_mean"), row.get("field_rmse_ci95"))
        lines.append(f"{name} & {mae} & {rmse} & {fmae} & {frmse} \\\\")
    lines += [
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-fold", default="results/loso_per_fold.csv")
    ap.add_argument("--summary", default="results/loso_summary.csv")
    ap.add_argument("--figdir", default="paper/figures")
    ap.add_argument("--tabledir", default="paper/tables")
    args = ap.parse_args()

    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.tabledir, exist_ok=True)

    summary = pd.read_csv(args.summary)
    per_fold = pd.read_csv(args.per_fold)

    _set_style()

    plot_mae_bars(summary, os.path.join(args.figdir, "loso_mae_bars.png"), metric="mae")
    plot_field_mae_bars(summary, os.path.join(args.figdir, "loso_field_mae_bars.png"))
    plot_per_fold_heatmap(per_fold, os.path.join(args.figdir, "loso_per_fold_heatmap.png"))
    plot_kan_comparison(summary, os.path.join(args.figdir, "kan_comparison.png"))
    write_latex_table(summary, os.path.join(args.tabledir, "loso_summary.tex"))


if __name__ == "__main__":
    main()
