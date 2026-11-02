# 0002 — Labeling LOSO run variants

- **Status:** accepted (2026-11-02)

## Context

`evaluation/loso_cross_validation.py` writes `results/loso_<Model>_<tag>.csv`
where `<tag>` encodes the configuration (`masked`, `unmasked`,
`seed42_aug_masked`, …), but the `model` column inside the CSV only contains the
architecture name. `evaluation/aggregate_loso.py` grouped by that column, so a
regularized rerun would have been pooled with its baseline (30 "folds").

## Decision

`aggregate_loso.py` derives the tag from the filename and:

- keeps the bare architecture name for the baseline tag `masked`
  (existing plots, `KAN_PAIRS` and paper tables keep working);
- labels every other run `<Model> (<tag>)`, e.g. `SimpleResNet (seed42_aug_masked)`;
- stores the raw values in new `base_model` and `variant` columns of
  `results/loso_per_fold.csv`.

## Consequences

Downstream consumers (`plot_loso_results.py`, `pairwise_stat_tests.py`) treat
each run as its own model. Tested in `tests/test_aggregate_loso.py`.
