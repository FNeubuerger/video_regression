import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.aggregate_loso import load_fold_csvs, summarise  # noqa: E402


def _write(path, model, maes):
    pd.DataFrame(
        {
            "fold": [f"US_{i:03d}" for i in range(len(maes))],
            "mae": maes,
            "rmse": maes,
            "model": model,
        }
    ).to_csv(path, index=False)


def test_variants_are_not_pooled(tmp_path):
    _write(tmp_path / "loso_SimpleResNet_masked.csv", "SimpleResNet", [10.0, 20.0])
    _write(
        tmp_path / "loso_SimpleResNet_seed42_aug_masked.csv",
        "SimpleResNet",
        [1.0, 3.0],
    )
    _write(tmp_path / "loso_SimpleResNet_unmasked.csv", "SimpleResNet", [5.0, 5.0])

    df = load_fold_csvs(str(tmp_path / "loso_*.csv"))
    summary = summarise(df).set_index("model")

    assert set(summary.index) == {
        "SimpleResNet",
        "SimpleResNet (seed42_aug_masked)",
        "SimpleResNet (unmasked)",
    }
    assert (summary["n_folds"] == 2).all()
    assert summary.loc["SimpleResNet", "mae_mean"] == 15.0
    assert summary.loc["SimpleResNet (seed42_aug_masked)", "mae_mean"] == 2.0
    assert set(df["base_model"]) == {"SimpleResNet"}


def test_model_name_with_underscore_keeps_variant(tmp_path):
    _write(tmp_path / "loso_Foo_Bar_masked.csv", "Foo_Bar", [1.0])
    df = load_fold_csvs(str(tmp_path / "loso_*.csv"))
    assert df["variant"].tolist() == ["masked"]
    assert df["model"].tolist() == ["Foo_Bar"]
