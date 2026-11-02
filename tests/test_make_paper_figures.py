import numpy as np
import pandas as pd
import pytest

from evaluation import make_paper_figures as mpf
from evaluation.pairwise_stat_tests import pairwise_wilcoxon


def _synthetic_per_fold(n_folds: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    folds = [f"US_{i:03d}_30W_10min" for i in range(1, n_folds // 2 + 1)] + [
        f"US_{i:03d}_40W_5min" for i in range(n_folds // 2 + 1, n_folds + 1)
    ]
    rows = []
    for model, offset in [("A", 10.0), ("B", 20.0), ("C", 40.0)]:
        for fold in folds:
            rows.append(
                {
                    "fold": fold,
                    "mae": offset + rng.normal(0, 1),
                    "base_model": model,
                    "variant": "masked",
                }
            )
    for fold in folds:
        rows.append(
            {
                "fold": fold,
                "mae": 8.0 + rng.normal(0, 1),
                "base_model": "A",
                "variant": "seed42_aug_masked",
            }
        )
    rows.append(
        {"fold": folds[0], "mae": 1.0, "base_model": "A", "variant": "partial_run"}
    )
    return pd.DataFrame(rows)


def test_holm_adjust_matches_reference():
    adjusted = mpf.holm_adjust(np.array([0.01, 0.04, 0.03, np.nan]))
    np.testing.assert_allclose(adjusted[:3], [0.03, 0.06, 0.06])
    assert np.isnan(adjusted[3])


def test_fold_helpers():
    assert mpf.fold_protocol("US_003_30W_10min") == mpf.PROTOCOL_LONG
    assert mpf.fold_protocol("US_013_40W_5min") == mpf.PROTOCOL_SWEEP
    assert mpf.fold_label("US_013_40W_5min") == "013 (40W/5m)"


def test_load_per_fold_drops_incomplete_runs(tmp_path):
    path = tmp_path / "per_fold.csv"
    _synthetic_per_fold().to_csv(path, index=False)
    df = mpf.load_per_fold(str(path), min_folds=8)
    assert "partial_run" not in set(df["variant"])
    matrix = mpf.baseline_matrix(df)
    assert list(matrix.columns) == ["A", "B", "C"]
    assert matrix.shape == (8, 3)


def test_main_writes_all_outputs(tmp_path, monkeypatch):
    per_fold = tmp_path / "per_fold.csv"
    _synthetic_per_fold().to_csv(per_fold, index=False)
    figdir, tabledir = tmp_path / "fig", tmp_path / "tab"
    monkeypatch.setattr(
        "sys.argv",
        [
            "make_paper_figures.py",
            "--per-fold",
            str(per_fold),
            "--figdir",
            str(figdir),
            "--tabledir",
            str(tabledir),
            "--min-folds",
            "8",
            "--pairwise-csv",
            str(tmp_path / "holm.csv"),
        ],
    )
    mpf.main()
    for name in [
        "loso_mae_distribution.png",
        "loso_regularization_paired.png",
        "loso_wilcoxon_heatmap.png",
        "loso_fold_difficulty.png",
    ]:
        assert (figdir / name).stat().st_size > 0
    vs_best = (tabledir / "loso_wilcoxon_vs_best.tex").read_text()
    assert "against A" in vs_best
    assert r"\label{tab:loso_per_fold}" in (tabledir / "loso_per_fold.tex").read_text()
    holm = pd.read_csv(tmp_path / "holm.csv", index_col=0)
    assert holm.shape == (3, 3)


def test_pairwise_wilcoxon_pairs_folds_per_model_pair():
    df = _synthetic_per_fold()
    df = df[df["variant"] != "seed42_aug_masked"]
    df["model"] = df["base_model"] + "/" + df["variant"]
    p = pairwise_wilcoxon(df, "mae")
    # The 1-fold partial run must not erase the folds of the complete runs.
    assert p.loc["A/masked", "C/masked"] == pytest.approx(2 * 0.5**8)
