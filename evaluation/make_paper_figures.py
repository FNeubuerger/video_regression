"""Generate additional paper figures and tables from the LOSO per-fold results.

Consumes ``results/loso_per_fold.csv`` (written by ``aggregate_loso.py``) and
produces fold-level analyses that complement ``plot_loso_results.py``:

Figures (``paper/figures/``):
  - ``loso_mae_distribution.png``    Per-fold MAE distribution per model (log scale)
  - ``loso_regularization_paired.png`` Baseline vs. regularized MAE per held-out fold
  - ``loso_wilcoxon_heatmap.png``    Holm-adjusted pairwise Wilcoxon p-values
  - ``loso_fold_difficulty.png``     Per-fold MAE across models, by protocol

Tables (``paper/tables/``):
  - ``loso_robust_stats.tex``        Median/IQR, mean/SD, worst fold, mean rank,
                                     MAE per acquisition protocol
  - ``loso_wilcoxon_vs_best.tex``    Paired tests of every model vs. the best one
  - ``loso_per_fold.tex``            Full per-fold MAE matrix (supplementary)

Run from the repository root:
    .venv/bin/python evaluation/make_paper_figures.py
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare, wilcoxon

BASELINE_VARIANT = "masked"
REGULARIZED_VARIANT = "seed42_aug_masked"
DIVERGENT_MODELS = {"KANResNet"}
PROTOCOL_LONG = "30 W, 10 min"
PROTOCOL_SWEEP = "25--50 W, 5 min"


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["legend.fontsize"] = 8


def fold_protocol(fold: str) -> str:
    """Return the acquisition protocol encoded in a fold name like US_001_30W_10min."""
    match = re.search(r"_(\d+)min$", fold)
    if match and match.group(1) == "10":
        return PROTOCOL_LONG
    return PROTOCOL_SWEEP


def fold_label(fold: str) -> str:
    """Shorten US_012_35W_5min to '012 (35W/5m)'."""
    match = re.match(r"US_(\d+)_(\d+)W_(\d+)min", fold)
    if not match:
        return fold
    idx, power, minutes = match.groups()
    return f"{idx} ({power}W/{minutes}m)"


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment; NaNs are passed through."""
    pvals = np.asarray(pvals, dtype=float)
    adjusted = np.full_like(pvals, np.nan)
    valid = np.flatnonzero(~np.isnan(pvals))
    if valid.size == 0:
        return adjusted
    order = valid[np.argsort(pvals[valid])]
    m = valid.size
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        adjusted[idx] = running
    return adjusted


def _paired_wilcoxon(a: pd.Series, b: pd.Series) -> float:
    paired = pd.concat([a, b], axis=1).dropna()
    if len(paired) < 2 or np.allclose(paired.iloc[:, 0], paired.iloc[:, 1]):
        return float("nan")
    return float(wilcoxon(paired.iloc[:, 0], paired.iloc[:, 1]).pvalue)


def load_per_fold(path: str, min_folds: int) -> pd.DataFrame:
    """Load per-fold results, keeping only (model, variant) runs with all folds."""
    df = pd.read_csv(path)
    if "base_model" not in df.columns:
        df["base_model"] = df["model"]
    if "variant" not in df.columns:
        df["variant"] = BASELINE_VARIANT
    df = df.dropna(subset=["mae"])
    counts = df.groupby(["base_model", "variant"])["fold"].nunique()
    complete = counts[counts >= min_folds].index
    dropped = counts[counts < min_folds]
    for (model, variant), n in dropped.items():
        print(f"[make_paper_figures] skipping incomplete run {model}/{variant}: {n}")
    return df.set_index(["base_model", "variant"]).loc[complete].reset_index()


def baseline_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Folds x models MAE matrix for the baseline variant, ordered by median MAE."""
    base = df[df["variant"] == BASELINE_VARIANT]
    pivot = base.pivot_table(
        index="fold", columns="base_model", values="mae", aggfunc="mean"
    )
    return pivot[pivot.median().sort_values().index]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_mae_distribution(matrix: pd.DataFrame, out_path: str) -> None:
    long = matrix.reset_index().melt(id_vars="fold", var_name="model", value_name="mae")
    long["protocol"] = long["fold"].map(fold_protocol)
    order = list(matrix.columns)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    sns.boxplot(
        data=long,
        x="model",
        y="mae",
        order=order,
        color="#dddddd",
        width=0.6,
        fliersize=0,
        linewidth=0.8,
        ax=ax,
    )
    sns.stripplot(
        data=long,
        x="model",
        y="mae",
        hue="protocol",
        order=order,
        hue_order=[PROTOCOL_LONG, PROTOCOL_SWEEP],
        palette=["#1f77b4", "#ff7f0e"],
        size=3.5,
        jitter=0.18,
        alpha=0.85,
        ax=ax,
    )
    ax.set_yscale("log")
    floor = 3.0
    ax.set_ylim(bottom=floor * 0.8)
    for x, model in enumerate(order):
        for _fold, value in matrix[model][matrix[model] < floor].items():
            ax.plot(x, floor, marker="v", color="#d62728", markersize=6, zorder=5)
            ax.annotate(
                f"{value:.2f} K",
                xy=(x, floor),
                xytext=(x + 0.12, floor * 1.2),
                fontsize=6,
                color="#d62728",
            )
    ax.set_xlabel("")
    ax.set_ylabel("Per-fold MAE (K, log scale)")
    ax.set_title("LOSO per-fold MAE distribution (15 held-out sequences)")
    ax.tick_params(axis="x", labelrotation=40)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax.legend(
        title="Held-out protocol",
        loc="upper left",
        frameon=True,
        labels=[PROTOCOL_LONG.replace("--", "-"), PROTOCOL_SWEEP.replace("--", "-")],
        handles=ax.get_legend_handles_labels()[0][:2],
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_regularization_paired(df: pd.DataFrame, out_path: str) -> list[dict]:
    """Slope plot of baseline vs. regularized MAE per fold; returns per-model stats."""
    reg_models = sorted(
        df.loc[df["variant"] == REGULARIZED_VARIANT, "base_model"].unique(),
        key=lambda m: (m in DIVERGENT_MODELS, m),
    )
    if not reg_models:
        print("No regularized runs found — skipping paired regularization plot.")
        return []

    fig, axes = plt.subplots(
        1, len(reg_models), figsize=(2.0 * len(reg_models) + 0.6, 3.4)
    )
    axes = np.atleast_1d(axes)
    stats = []
    for ax, model in zip(axes, reg_models):
        sub = df[df["base_model"] == model].pivot_table(
            index="fold", columns="variant", values="mae"
        )
        sub = sub[[BASELINE_VARIANT, REGULARIZED_VARIANT]].dropna()
        before, after = sub[BASELINE_VARIANT], sub[REGULARIZED_VARIANT]
        improved = after < before
        for fold, b, a, better in zip(sub.index, before, after, improved):
            ax.plot(
                [0, 1],
                [b, a],
                color="#2ca02c" if better else "#d62728",
                alpha=0.7,
                linewidth=1.0,
                marker="o",
                markersize=3,
            )
        ax.plot(
            [0, 1],
            [before.median(), after.median()],
            color="black",
            linewidth=2.2,
            marker="s",
            markersize=5,
            zorder=4,
        )
        p = _paired_wilcoxon(before, after)
        stats.append(
            {
                "model": model,
                "improved": int(improved.sum()),
                "n": len(sub),
                "median_delta": float((after - before).median()),
                "p": p,
            }
        )
        if (sub.max().max() / max(sub.min().min(), 1e-6)) > 20:
            ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["baseline", "reg."])
        ax.set_xlim(-0.25, 1.25)
        ax.set_title(
            f"{model}\n{int(improved.sum())}/{len(sub)} folds improve, $p$={p:.3f}",
            fontsize=9,
        )
    axes[0].set_ylabel("Per-fold MAE (K)")
    fig.legend(
        handles=[
            Line2D([0], [0], color="#2ca02c", label="fold improves"),
            Line2D([0], [0], color="#d62728", label="fold degrades"),
            Line2D([0], [0], color="black", marker="s", lw=2.2, label="median"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Weight decay + augmentation vs. baseline, per held-out sequence", fontsize=11
    )
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")
    return stats


def pairwise_holm(matrix: pd.DataFrame) -> pd.DataFrame:
    models = list(matrix.columns)
    n = len(models)
    raw = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            raw[i, j] = _paired_wilcoxon(matrix[models[i]], matrix[models[j]])
    iu = np.triu_indices(n, k=1)
    adjusted = np.full((n, n), np.nan)
    adjusted[iu] = holm_adjust(raw[iu])
    adjusted = np.where(np.isnan(adjusted), adjusted.T, adjusted)
    np.fill_diagonal(adjusted, 1.0)
    return pd.DataFrame(adjusted, index=models, columns=models)


def plot_wilcoxon_heatmap(p_holm: pd.DataFrame, out_path: str) -> None:
    # Drop the always-empty first row / last column of the lower triangle.
    p_holm = p_holm.iloc[1:, :-1]
    mask = np.triu(np.ones_like(p_holm, dtype=bool), k=1)

    def _p_label(p: float) -> str:
        if p < 0.001:
            return "<.001"
        return (f"{p:.3f}" if p < 0.01 else f"{p:.2f}").lstrip("0")

    annot = p_holm.map(_p_label)

    n = len(p_holm)
    fig, ax = plt.subplots(figsize=(0.55 * n + 2.2, 0.5 * n + 1.6))
    ax.grid(False)
    sns.heatmap(
        -np.log10(p_holm.clip(lower=1e-4)),
        mask=mask,
        annot=annot,
        fmt="",
        cmap="rocket_r",
        vmin=0,
        vmax=4,
        linewidths=0.4,
        linecolor="white",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": r"$-\log_{10}\,p_\mathrm{Holm}$"},
        ax=ax,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.axhline(-np.log10(0.05), color="black", linewidth=1.0)
    ax.set_title("Pairwise Wilcoxon signed-rank tests on per-fold MAE (Holm)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=40)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fold_difficulty(matrix: pd.DataFrame, out_path: str) -> None:
    stable = matrix.drop(columns=[m for m in DIVERGENT_MODELS if m in matrix])
    folds = sorted(stable.index)
    long = stable.loc[folds].reset_index().melt(id_vars="fold", value_name="mae")
    long["label"] = long["fold"].map(fold_label)
    long["protocol"] = long["fold"].map(fold_protocol)
    labels = [fold_label(f) for f in folds]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    sns.boxplot(
        data=long,
        x="label",
        y="mae",
        order=labels,
        hue="protocol",
        hue_order=[PROTOCOL_LONG, PROTOCOL_SWEEP],
        palette=["#aec7e8", "#ffbb78"],
        dodge=False,
        width=0.6,
        fliersize=2,
        linewidth=0.8,
        ax=ax,
    )
    best = stable.loc[folds].min(axis=1).to_numpy()
    ax.plot(
        range(len(folds)), best, "k_", markersize=10, mew=2, label="best model on fold"
    )
    ymax = 180.0
    ax.set_ylim(0, ymax)
    for x, fold in enumerate(folds):
        clipped = stable.loc[fold][stable.loc[fold] > ymax]
        for model, value in clipped.items():
            ax.annotate(
                f"{model}\n{value:.0f} K",
                xy=(x, ymax),
                xytext=(x + 0.3, ymax * 0.72),
                fontsize=6,
                ha="left",
                arrowprops={"arrowstyle": "->", "lw": 0.6},
            )
    ax.set_xlabel("Held-out sequence")
    ax.set_ylabel("MAE across models (K)")
    ax.set_title(
        f"Fold difficulty: per-fold MAE of {stable.shape[1]} models"
        f" ({', '.join(sorted(DIVERGENT_MODELS))} excluded)"
    )
    ax.tick_params(axis="x", labelrotation=55)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [lbl.replace("--", "-") for lbl in lbls],
        loc="upper right",
        frameon=True,
        ncol=3,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _fmt(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "--"
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", "{,}")
    return f"{value:.{digits}f}"


def _write(lines: list[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def write_robust_table(matrix: pd.DataFrame, out_path: str) -> None:
    ranks = matrix.rank(axis=1).mean()
    friedman_p = friedmanchisquare(*[matrix[c] for c in matrix.columns]).pvalue
    protocol = matrix.index.map(fold_protocol)
    long_mask = np.asarray(protocol == PROTOCOL_LONG)
    n_long, n_sweep = int(long_mask.sum()), int((~long_mask).sum())
    friedman_str = f"{friedman_p:.1e}".replace("e-0", r"\times10^{-") + "}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Robust LOSO statistics on per-fold MAE (K), ordered by median. "
        r"IQR = interquartile range; worst = largest single-fold MAE; rank = mean "
        r"rank across the 15 folds (1 = best; Friedman test "
        rf"$p={friedman_str}$). The last two columns split the folds by acquisition "
        rf"protocol: {n_long} sequences at {PROTOCOL_LONG} and {n_sweep} sequences "
        rf"at {PROTOCOL_SWEEP}.}}",
        r"\label{tab:loso_robust}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Median} [IQR] & \textbf{Mean} $\pm$ SD "
        r"& \textbf{Worst} & \textbf{Rank} "
        rf"& \textbf{{{PROTOCOL_LONG}}} & \textbf{{{PROTOCOL_SWEEP}}} \\",
        r"\midrule",
    ]
    best_rank = ranks.min()
    for model in matrix.columns:
        col = matrix[model]
        q1, q3 = col.quantile([0.25, 0.75])
        rank = f"{ranks[model]:.1f}"
        if np.isclose(ranks[model], best_rank):
            rank = rf"\textbf{{{rank}}}"
        lines.append(
            f"{latex_escape(model)} & {_fmt(col.median())} [{_fmt(q1)}--{_fmt(q3)}] "
            f"& {_fmt(col.mean())} $\\pm$ {_fmt(col.std())} & {_fmt(col.max())} "
            f"& {rank} & {_fmt(col[long_mask].mean())} "
            f"& {_fmt(col[~long_mask].mean())} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    _write(lines, out_path)


def write_vs_best_table(matrix: pd.DataFrame, out_path: str) -> None:
    best = matrix.columns[0]
    others = list(matrix.columns[1:])
    raw = np.array([_paired_wilcoxon(matrix[best], matrix[m]) for m in others])
    adjusted = holm_adjust(raw)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Paired comparison of every model against {latex_escape(best)}, "
        r"the model with the lowest median LOSO MAE. $\Delta$ is the median per-fold "
        r"MAE difference (model $-$ reference, K); wins counts folds on which the "
        r"model beats the reference. $p$ is the two-sided Wilcoxon signed-rank test "
        r"over the 15 folds; $p_\mathrm{Holm}$ is Holm-adjusted for "
        rf"{len(others)} comparisons.}}",
        r"\label{tab:loso_vs_best}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{$\Delta$ MAE} (K) & \textbf{Wins} & $p$ "
        r"& $p_\mathrm{Holm}$ \\",
        r"\midrule",
    ]
    for model, p, p_adj in zip(others, raw, adjusted):
        diff = matrix[model] - matrix[best]
        wins = int((diff < 0).sum())
        p_raw = f"{p:.3f}" if p >= 0.001 else "$<$0.001"
        p_holm = f"{p_adj:.3f}" if p_adj >= 0.001 else "$<$0.001"
        if p_adj < 0.05:
            p_holm = rf"\textbf{{{p_holm}}}"
        lines.append(
            f"{latex_escape(model)} & {diff.median():+.1f} & {wins}/{len(diff)} "
            f"& {p_raw} & {p_holm} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    _write(lines, out_path)


def write_per_fold_table(matrix: pd.DataFrame, out_path: str) -> None:
    models = list(matrix.columns)
    folds = sorted(matrix.index)
    header = " & ".join(rf"\rotatebox{{70}}{{{latex_escape(m)}}}" for m in models)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-fold LOSO MAE (K) for every model (baseline configuration). "
        r"Each row is one held-out sequence (index, ablation power, duration); the "
        r"best model per fold is shown in bold.}",
        r"\label{tab:loso_per_fold}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{l{'r' * len(models)}}}",
        r"\toprule",
        rf"\textbf{{Held-out}} & {header} \\",
        r"\midrule",
    ]
    for fold in folds:
        row = matrix.loc[fold]
        best = row.idxmin()
        cells = []
        for model in models:
            cell = _fmt(row[model])
            cells.append(rf"\textbf{{{cell}}}" if model == best else cell)
        lines.append(f"{fold_label(fold)} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(
        "Median & " + " & ".join(_fmt(matrix[m].median()) for m in models) + r" \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    _write(lines, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-fold", default="results/loso_per_fold.csv")
    ap.add_argument("--figdir", default="paper/figures")
    ap.add_argument("--tabledir", default="paper/tables")
    ap.add_argument("--min-folds", type=int, default=15)
    ap.add_argument(
        "--pairwise-csv",
        default="results/pairwise_wilcoxon_holm.csv",
        help="Where to write the Holm-adjusted pairwise p-value matrix.",
    )
    args = ap.parse_args()

    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.tabledir, exist_ok=True)
    _set_style()

    df = load_per_fold(args.per_fold, args.min_folds)
    matrix = baseline_matrix(df)

    plot_mae_distribution(
        matrix, os.path.join(args.figdir, "loso_mae_distribution.png")
    )
    plot_regularization_paired(
        df, os.path.join(args.figdir, "loso_regularization_paired.png")
    )
    p_holm = pairwise_holm(matrix)
    p_holm.to_csv(args.pairwise_csv)
    print(f"Wrote {args.pairwise_csv}")
    plot_wilcoxon_heatmap(
        p_holm, os.path.join(args.figdir, "loso_wilcoxon_heatmap.png")
    )
    plot_fold_difficulty(matrix, os.path.join(args.figdir, "loso_fold_difficulty.png"))

    write_robust_table(matrix, os.path.join(args.tabledir, "loso_robust_stats.tex"))
    write_vs_best_table(
        matrix, os.path.join(args.tabledir, "loso_wilcoxon_vs_best.tex")
    )
    write_per_fold_table(matrix, os.path.join(args.tabledir, "loso_per_fold.tex"))


if __name__ == "__main__":
    main()
