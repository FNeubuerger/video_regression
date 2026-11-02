# Current Results

Last updated: 2026-11-02 (commit `698f296`, "added some recent results").
This file is the single place for headline numbers. Raw per-fold CSVs live in
[`results/`](../results/); paper tables in [`paper/tables/`](../paper/tables/).

## Evaluation protocol

- **LOSO (Leave-One-Sequence-Out):** 15 phantom sequences (`US_001` … `US_015`),
  each held out once. This is the protocol the paper's claims rest on.
- All LOSO runs: 10 epochs, batch size 32, thermometer artifact masking
  (`--masked`), scalar MAE/RMSE in Kelvin on the held-out sequence.
- Mixed/random-split numbers (e.g. "BioheatPINN 0.26 K", "SimpleResNet 1.31 K")
  leak subject identity between train and test and are **not** comparable to LOSO.

## LOSO baseline (no weight decay, no augmentation)

Source: `results/loso_<Model>_masked.csv`, aggregated in `results/loso_summary.csv`.
Mean ± 95 % CI over 15 folds.

| Model | MAE (K) | RMSE (K) | Notes |
| :--- | ---: | ---: | :--- |
| SpatialResNet | 23.08 ± 5.05 | 25.93 ± 4.75 | best mean MAE |
| CNNLSTM | 23.71 ± 4.05 | 27.25 ± 4.18 | |
| BayesianCNNLSTM | 25.68 ± 4.25 | 30.48 ± 4.68 | |
| SimpleResNet | 26.67 ± 5.14 | 30.94 ± 6.03 | |
| FullBayesianResNet | 32.09 ± 13.36 | 36.68 ± 13.31 | |
| PretrainedCNNLSTM | 32.10 ± 5.91 | 36.89 ± 5.96 | |
| PhysicsCNNLSTM | 34.68 ± 7.35 | 40.08 ± 7.15 | |
| SpatialKANBioheat | 47.78 ± 11.82 | 49.60 ± 12.09 | |
| ConvectionBioheat | 49.85 ± 10.42 | 51.64 ± 10.72 | field MAE 0.05 K is suspiciously low, see TODO |
| BayesianResNet | 55.82 ± 29.65 | 61.92 ± 30.15 | a few folds diverge (US_005: 237 K) |
| ConvLTC | 97.14 ± 19.14 | 98.30 ± 19.26 | |
| KANResNet | 3359 ± 3504 | 3385 ± 3533 | diverges, numerically unstable |

## LOSO regularized (weight decay 1e-4 + flip/brightness augmentation + seed 42)

Source: `results/loso_<Model>_seed42_aug_masked.csv` (committed 2026-11-02).
Paper table: [`paper/tables/loso_regularized.tex`](../paper/tables/loso_regularized.tex).
`p` = paired two-sided Wilcoxon signed-rank on per-fold MAE vs. baseline.

| Model | MAE (K) | Baseline MAE (K) | Δ MAE | Folds improved | p |
| :--- | ---: | ---: | ---: | ---: | ---: |
| SimpleResNet | 22.88 ± 4.59 | 26.67 ± 5.14 | −3.79 | 10/15 | 0.135 |
| CNNLSTM | 31.15 ± 6.59 | 23.71 ± 4.05 | **+7.44** | 5/15 | **0.026** |
| BayesianResNet | 75.66 ± 50.59 | 55.82 ± 29.65 | +19.84 | 9/15 | 0.978 |
| KANResNet | 3414 ± 2851 | 3359 ± 3504 | +55.51 | 6/15 | 0.720 |

### Interpretation

1. **Regularization does not close the generalization gap.** The best regularized
   model (SimpleResNet, 22.9 K) is on par with the best unregularized one
   (SpatialResNet, 23.1 K). Every image-only scalar model stays above 20 K MAE.
2. **The effect is architecture-dependent:** small, non-significant gain for
   SimpleResNet; significant degradation for CNNLSTM. Weight decay and
   augmentation were changed together and only one seed was run, so the two
   effects (and seed variance) are confounded.
3. **BayesianResNet** median MAE is ~33 K, but single folds blow up
   (US_006: 395 K, US_008: 184 K), so its mean is not meaningful.
4. **KANResNet** diverges regardless of regularization (median ~700 K).
5. `results/loso_SimpleResNet_seed123_aug_unmasked.csv` contains a single fold
   (US_001: 26.0 K) from an aborted run and should not be reported.

## Caveats

- Regularized LOSO checkpoints overwrote the baseline ones in
  `checkpoints/loso/<Model>/` (same file names), so the baseline checkpoints for
  SimpleResNet, CNNLSTM, BayesianResNet and KANResNet are gone; only the CSVs remain.
- All LOSO runs were produced on host `dsfhswf-vinson` (git sha `ece528b` for
  the regularized runs, see `checkpoints/loso/<Model>/fold_*_config.json`).

## Reproducing

```bash
make loso_aggregate   # results/loso_summary.csv + results/loso_per_fold.csv
make stat_tests       # pairwise Wilcoxon matrix
make loso_plots       # paper figures + paper/tables/loso_summary.tex
```
