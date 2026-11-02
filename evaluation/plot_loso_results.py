"""Generate baseline LOSO figures and a LaTeX summary table.

Consumes:
  - ``results/loso_per_fold.csv`` (long format, one row per (model, fold))
  - ``results/loso_summary.csv``  (one row per run variant, with mean/std/sem/CI95)

The paper-facing summary figures and table use only complete (15-fold) baseline
masked runs. Regularized variants are reported separately, and incomplete runs
remain in the aggregate CSVs for provenance but are excluded from these outputs.

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
import sys

# Allow `import models.*` when invoked from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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
    means = df[col_mean].to_numpy()
    ci = df[col_ci].fillna(0).to_numpy()
    floor = 1.0
    lower = np.maximum(means - ci, floor)
    ax.bar(
        np.arange(len(df)),
        means - floor,
        bottom=floor,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.errorbar(
        np.arange(len(df)),
        means,
        yerr=np.vstack([means - lower, ci]),
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=0.8,
    )
    ax.set_yscale("log")
    ax.set_ylim(floor, max(means + ci) * 1.25)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["model"])
    ax.set_ylabel(
        f"{metric.upper()} (K, log scale) — mean $\\pm$ 95% CI across 15 LOSO folds"
    )
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
    means = df["field_mae_mean"].to_numpy()
    ci = df["field_mae_ci95"].fillna(0).to_numpy()
    floor = 0.01
    lower = np.maximum(means - ci, floor)
    ax.bar(
        np.arange(len(df)),
        means - floor,
        bottom=floor,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.errorbar(
        np.arange(len(df)),
        means,
        yerr=np.vstack([means - lower, ci]),
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=0.8,
    )
    ax.set_yscale("log")
    ax.set_ylim(floor, max(means + ci) * 1.25)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["model"])
    ax.set_ylabel("Field MAE (K, log scale)")
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
    labels = pivot.map(lambda value: "<0.1" if value < 0.1 else f"{value:.1f}")
    sns.heatmap(
        pivot.clip(lower=10, upper=150),
        annot=labels,
        fmt="",
        cmap="viridis",
        vmin=10,
        vmax=150,
        cbar_kws={"label": "MAE (K, color clipped to 10–150)"},
        ax=ax,
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_title("Per-fold MAE across LOSO splits (color clipped for readability)")
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def _count_params(model_name: str) -> float:
    """Return trainable parameter count (millions) for a model, or NaN on failure.

    Instantiates the model on CPU using the same constructors as the LOSO factory.
    Results are cached per-process.
    """
    cache = _count_params._cache  # type: ignore[attr-defined]
    if model_name in cache:
        return cache[model_name]

    import importlib

    frame_shape = (64, 64, 5)
    time_steps = 5
    try:
        if model_name == "CNNLSTM":
            m = importlib.import_module("models.backbones").CNNLSTM(
                frame_shape=frame_shape, time_steps=time_steps
            )
        elif model_name == "PretrainedCNNLSTM":
            import torch.nn as nn
            from torchvision.models import resnet18

            backbone = resnet18()
            backbone.fc = nn.Linear(512, 1)
            m = importlib.import_module("models.backbones").PretrainedCNNLSTM(
                backbone, frame_shape=frame_shape, time_steps=time_steps
            )
        elif model_name == "SimpleResNet":
            m = importlib.import_module("models.backbones").SimpleResNet(
                frame_shape=frame_shape
            )
        elif model_name == "SpatialResNet":
            m = importlib.import_module("models.backbones").SpatialResNet(
                frame_shape=frame_shape
            )
        elif model_name == "PhysicsCNNLSTM":
            m = importlib.import_module("physics.models").PhysicsCNNLSTM(
                frame_shape=frame_shape, time_steps=time_steps, pretrained=False
            )
        elif model_name == "ConvectionBioheat":
            m = importlib.import_module("physics.models").SpatialPhysicsCNNLSTM(
                frame_shape=frame_shape, time_steps=time_steps, pretrained=False
            )
        elif model_name == "BayesianResNet":
            m = importlib.import_module("models.bayesian").BayesianResNet(
                frame_shape=frame_shape
            )
        elif model_name == "FullBayesianResNet":
            m = importlib.import_module("models.bayesian").FullBayesianResNet(
                frame_shape=frame_shape
            )
        elif model_name == "BayesianCNNLSTM":
            m = importlib.import_module("models.bayesian").BayesianCNNLSTM(
                frame_shape=frame_shape
            )
        elif model_name == "ConvLTC":
            m = importlib.import_module("models.conv_ltc").ConvLTC(
                in_channels=5, hidden_channels=32
            )
        elif model_name == "KANResNet":
            m = importlib.import_module("models.kan").KANResNet(frame_shape=frame_shape)
        elif model_name == "SpatialKANBioheat":
            m = importlib.import_module("models.kan").SpatialKANBioheat(
                frame_shape=frame_shape, time_steps=time_steps, output_hw=(4, 4)
            )
        else:
            cache[model_name] = float("nan")
            return cache[model_name]
        n = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
        cache[model_name] = n
    except Exception as exc:  # pragma: no cover — best-effort param count
        print(f"[plot_loso] could not count params for {model_name}: {exc}")
        cache[model_name] = float("nan")
    return cache[model_name]


_count_params._cache = {}  # type: ignore[attr-defined]


def _paired_per_fold_panel(ax, per_fold: pd.DataFrame) -> None:
    """Per-fold paired MAE — for each KAN pair, baseline → KAN per held-out subject."""
    from scipy.stats import wilcoxon

    pairs_present = []
    for baseline, kan in KAN_PAIRS:
        b = per_fold[per_fold["model"] == baseline][["fold", "mae"]].rename(
            columns={"mae": "baseline"}
        )
        k = per_fold[per_fold["model"] == kan][["fold", "mae"]].rename(
            columns={"mae": "kan"}
        )
        merged = b.merge(k, on="fold", how="inner").dropna()
        if merged.empty:
            continue
        pairs_present.append((baseline, kan, merged))

    if not pairs_present:
        ax.text(
            0.5,
            0.5,
            "KAN per-fold results not available yet\n(runs still in progress)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="grey",
        )
        ax.set_axis_off()
        return

    # x positions: one slot per pair, two columns inside
    pair_xs = np.arange(len(pairs_present))
    width = 0.35
    for idx, (baseline, kan, merged) in enumerate(pairs_present):
        xb = np.full(len(merged), pair_xs[idx] - width / 2)
        xk = np.full(len(merged), pair_xs[idx] + width / 2)
        # Lines per subject
        for j in range(len(merged)):
            colour = (
                "#2ca02c"
                if merged["kan"].iloc[j] < merged["baseline"].iloc[j]
                else "#d62728"
            )
            ax.plot(
                [xb[j], xk[j]],
                [merged["baseline"].iloc[j], merged["kan"].iloc[j]],
                color=colour,
                alpha=0.6,
                linewidth=1.0,
            )
        ax.scatter(
            xb, merged["baseline"], color="#1f77b4", s=30, zorder=3, label="_baseline"
        )
        ax.scatter(xk, merged["kan"], color="#9467bd", s=30, zorder=3, label="_kan")
        # Wilcoxon p-value (paired, two-sided)
        try:
            stat, p = wilcoxon(merged["baseline"], merged["kan"])
            pstr = f"p={p:.3f}" if p >= 0.001 else f"p={p:.1e}"
        except ValueError:
            pstr = "p=n/a"
        mean_delta = (merged["baseline"] - merged["kan"]).mean()
        ax.text(
            pair_xs[idx],
            ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else merged.max().max() * 1.05,
            f"$\\Delta$={mean_delta:+.2f} K\n{pstr}",
            ha="center",
            va="top",
            fontsize=9,
        )

    ax.set_xticks(pair_xs)
    ax.set_xticklabels([f"{b}\n$\\to$\n{k}" for b, k, _ in pairs_present], fontsize=9)
    ax.set_ylabel("Per-subject MAE (K)")
    if all(
        merged[["baseline", "kan"]].min().min() > 0 for _, _, merged in pairs_present
    ):
        ax.set_yscale("log")
        ax.set_ylabel("Per-subject MAE (K, log scale)")
    ax.set_title("Paired per-fold comparison (green = KAN wins on that subject)")
    # Build a custom legend
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#1f77b4",
                label="Baseline",
            ),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="#9467bd", label="KAN"
            ),
            Line2D([0], [0], color="#2ca02c", label="KAN improves"),
            Line2D([0], [0], color="#d62728", label="KAN regresses"),
        ],
        loc="lower left",
        fontsize=8,
    )


def _pareto_panel(ax, summary: pd.DataFrame) -> None:
    """MAE vs trainable parameter count — KANs highlighted, Pareto frontier drawn."""
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            {
                "Model": r["model"],
                "MAE": r["mae_mean"],
                "MAE_CI": r["mae_ci95"],
                "Params_M": _count_params(r["model"]),
                "is_KAN": "KAN" in r["model"],
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["MAE", "Params_M"])
    if df.empty:
        ax.set_axis_off()
        return

    # Pareto frontier (minimize MAE as params grow).
    df_sorted = df.sort_values("Params_M")
    pareto_mask, best = [], float("inf")
    for v in df_sorted["MAE"]:
        if v < best:
            pareto_mask.append(True)
            best = v
        else:
            pareto_mask.append(False)
    pareto = df_sorted[pareto_mask]
    ax.plot(
        pareto["Params_M"],
        pareto["MAE"],
        color="#888888",
        linestyle="--",
        linewidth=1.2,
        zorder=1,
        label="Pareto front",
    )

    base = df[~df["is_KAN"]]
    kan = df[df["is_KAN"]]
    base_lower = np.maximum(base["MAE"].to_numpy() - base["MAE_CI"].to_numpy(), 1)
    ax.errorbar(
        base["Params_M"],
        base["MAE"],
        yerr=np.vstack(
            [base["MAE"].to_numpy() - base_lower, base["MAE_CI"].to_numpy()]
        ),
        fmt="o",
        color="#4c72b0",
        ecolor="#4c72b0",
        alpha=0.85,
        capsize=2,
        markersize=5,
        linewidth=1.0,
        label="Baselines",
        zorder=2,
    )
    if not kan.empty:
        kan_lower = np.maximum(kan["MAE"].to_numpy() - kan["MAE_CI"].to_numpy(), 1)
        ax.errorbar(
            kan["Params_M"],
            kan["MAE"],
            yerr=np.vstack(
                [kan["MAE"].to_numpy() - kan_lower, kan["MAE_CI"].to_numpy()]
            ),
            fmt="D",
            color="#c44e52",
            ecolor="#c44e52",
            capsize=3,
            markersize=8,
            linewidth=1.4,
            label="KAN variants",
            zorder=4,
        )

    label_offsets = {
        "ConvLTC": (8, 8),
        "CNNLSTM": (-65, 12),
        "SpatialResNet": (8, -14),
        "KANResNet": (-65, 8),
        "SpatialKANBioheat": (8, 8),
    }
    for _, row in df.iterrows():
        if row["Model"] not in label_offsets:
            continue
        colour = "#c44e52" if row["is_KAN"] else "#222222"
        ax.annotate(
            row["Model"],
            (row["Params_M"], row["MAE"]),
            xytext=label_offsets[row["Model"]],
            textcoords="offset points",
            fontsize=7,
            color=colour,
            fontweight="bold" if row["is_KAN"] else "normal",
            arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.5},
        )

    # Reference: best baseline MAE
    best_baseline = base["MAE"].min() if not base.empty else None
    if best_baseline is not None:
        ax.axhline(
            best_baseline,
            color="#4c72b0",
            linestyle=":",
            linewidth=0.8,
            alpha=0.6,
            label=f"Best baseline ({best_baseline:.1f} K)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Trainable parameters (M, log scale)")
    ax.set_ylabel("LOSO MAE (K, log scale) — mean $\\pm$ 95% CI")
    ax.set_title("Parameter-efficiency Pareto frontier")
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, which="both", alpha=0.3)


def plot_kan_comparison(
    summary: pd.DataFrame, per_fold: pd.DataFrame, out_path: str
) -> None:
    """KAN-vs-baseline comparison.

    Layout adapts to data availability:
      * If KAN per-fold data exists, draw two panels:
          - Left: paired within-subject MAE (baseline -> KAN) with Wilcoxon p.
          - Right: parameter-efficiency Pareto (log-log).
      * Otherwise, draw the Pareto panel full width so the figure still conveys
        the parameter-efficiency landscape of the baseline ladder.
    """
    kan_has_data = any(
        not per_fold[per_fold["model"] == kan].empty for _, kan in KAN_PAIRS
    )

    if kan_has_data:
        fig, axes = plt.subplots(
            1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.05, 1.3]}
        )
        _paired_per_fold_panel(axes[0], per_fold)
        _pareto_panel(axes[1], summary)
        fig.suptitle(
            "KAN vs baseline — within-subject paired error and parameter efficiency",
            fontsize=12,
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
        _pareto_panel(ax, summary)
        # Avoid duplicate title; replace the panel title with the figure-level one.
        ax.set_title("")
        fig.suptitle(
            "Parameter-efficiency landscape (KAN runs in progress)",
            fontsize=12,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
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
        r"\caption{Baseline masked leave-one-subject-out cross-validation results "
        r"(15 folds). "
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


def select_complete_baseline_runs(
    per_fold: pd.DataFrame, summary: pd.DataFrame, min_folds: int = 15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only complete baseline-masked runs for the baseline paper plots/table."""
    per_fold = per_fold.copy()
    if "base_model" not in per_fold.columns:
        per_fold["base_model"] = per_fold["model"]
    if "variant" not in per_fold.columns:
        per_fold["variant"] = "masked"

    counts = per_fold.groupby(["base_model", "variant"])["fold"].nunique()
    baseline_counts = counts[counts.index.get_level_values("variant") == "masked"]
    complete_models = set(
        baseline_counts[baseline_counts >= min_folds].index.get_level_values(
            "base_model"
        )
    )
    for (model, variant), count in counts.items():
        if variant != "masked" or count < min_folds:
            print(
                f"Skipping non-baseline or incomplete run for summary plots: "
                f"{model} ({variant}), {count} folds"
            )

    baseline_folds = per_fold[
        (per_fold["variant"] == "masked") & per_fold["base_model"].isin(complete_models)
    ].copy()
    baseline_folds["model"] = baseline_folds["base_model"]
    baseline_summary = summary[
        summary["model"].isin(complete_models) & (summary["n_folds"] >= min_folds)
    ].copy()
    if baseline_summary.empty or baseline_folds.empty:
        raise ValueError(
            "No complete baseline masked LOSO runs found; "
            f"expected at least {min_folds} folds per model."
        )
    return baseline_folds, baseline_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-fold", default="results/loso_per_fold.csv")
    ap.add_argument("--summary", default="results/loso_summary.csv")
    ap.add_argument("--figdir", default="paper/figures")
    ap.add_argument("--tabledir", default="paper/tables")
    ap.add_argument("--min-folds", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.tabledir, exist_ok=True)

    per_fold = pd.read_csv(args.per_fold)
    summary = pd.read_csv(args.summary)
    per_fold, summary = select_complete_baseline_runs(
        per_fold, summary, min_folds=args.min_folds
    )

    _set_style()

    plot_mae_bars(summary, os.path.join(args.figdir, "loso_mae_bars.png"), metric="mae")
    plot_field_mae_bars(summary, os.path.join(args.figdir, "loso_field_mae_bars.png"))
    plot_per_fold_heatmap(
        per_fold, os.path.join(args.figdir, "loso_per_fold_heatmap.png")
    )
    plot_kan_comparison(
        summary, per_fold, os.path.join(args.figdir, "kan_comparison.png")
    )
    write_latex_table(summary, os.path.join(args.tabledir, "loso_summary.tex"))


if __name__ == "__main__":
    main()
