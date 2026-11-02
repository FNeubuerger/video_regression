# TODO

Single source of truth for open work (see [docs/decisions/0001](docs/decisions/0001-documentation-layout.md)).
Current numbers: [docs/RESULTS.md](docs/RESULTS.md). Last reviewed: 2026-11-02.

## 1. Results & science (highest priority)

The regularized LOSO runs (committed 2026-11-02) show that weight decay +
augmentation do **not** close the mixed-split vs. LOSO gap (best LOSO MAE still
~23 K). The paper narrative has been corrected accordingly.

- [ ] **Disentangle the regularization ablation.** Weight decay and augmentation
  were changed together with a single seed. Run SimpleResNet + CNNLSTM with
  (a) WD only, (b) aug only, (c) both, each with ≥3 seeds (42/123/7).
- [ ] **Investigate why CNNLSTM gets worse with augmentation** (+7.4 K, p=0.026).
  Suspect: horizontal flip + negated flow-dx is inconsistent with the temporal
  model, or brightness jitter destroys the intensity cue the LSTM relies on.
- [ ] **Root-cause the generalization gap** beyond regularization: per-sequence
  normalization / intensity calibration, domain shift between 10 min @ 30 W and
  5 min @ 25–50 W sequences (`US_010`–`US_015`), predicting ΔT instead of absolute T.
- [ ] **Stabilize or drop KANResNet** (MAE ~3400 K in both configs; diverges on
  `US_008`, `US_015`). Options: clip/normalize KAN inputs, spline-grid
  regularization, lower LR. Otherwise report only SpatialKANBioheat.
- [ ] **BayesianResNet fold blow-ups** (US_005: 237 K baseline; US_006: 395 K
  regularized). Check KL weighting / MC sample count; report median alongside mean.
- [ ] **Sanity-check ConvectionBioheat field MAE of 0.05 K** — implausibly low
  vs. 49.9 K scalar MAE; likely the field target equals the physics prior. Also
  ConvectionBioheat/SpatialKANBioheat score 0.009/0.043 K *scalar* MAE on fold
  `US_001` (all other models 15–45 K) — flagged as suspected artifact in the paper.
- [ ] Physics models degrade on the 5 min power sweep (ConvectionBioheat 42→61 K,
  SpatialKANBioheat 39→61 K); check whether the bioheat parameters are fixed to
  the 30 W protocol (see `tab:loso_robust`).
- [ ] Extend regularized LOSO to the spatial/physics models (SpatialResNet,
  ConvectionBioheat, SpatialKANBioheat) once the ablation above is settled.
- [ ] Re-run `make loso_plots` and update `paper/tables/loso_summary.tex` and the
  bar/heatmap figures (now that run variants are labeled separately, see
  [decision 0002](docs/decisions/0002-loso-run-variant-labels.md)); filter out
  incomplete runs first (the seed123 run would appear as a 1-fold row).
- [ ] Delete or finish `results/loso_SimpleResNet_seed123_aug_unmasked.csv`
  (aborted run, 1 fold only).

## 2. Reproducibility & infrastructure

- [ ] **Do not overwrite checkpoints between LOSO configurations.** Regularized
  runs overwrote `checkpoints/loso/<Model>/fold_*.pth` for SimpleResNet, CNNLSTM,
  BayesianResNet, KANResNet; include the run tag in the checkpoint directory.
- [ ] Clean up `results/MASTER_RESULTS.csv` (mixes legacy `comprehensive_results`
  rows, NLL of 1e14 for `best`, and LOSO columns).
- [ ] Data manifest: `data/DATA_MANIFEST.json` (power, duration, frame count,
  sensor positions per sequence) + checksums for `level1_cropped`.

## 3. Evaluation pipeline bugs (legacy evaluators; LOSO does not depend on them)

- [ ] `evaluation/evaluate_models.py`: shape mismatch (10776,) vs (43104,) —
  frame-level vs sequence-level predictions (×4). Reduce to sequence level or drop
  legacy models from the queue. Blocks `metrics_comparison.csv`,
  `paper/viz_performance.py`, `evaluation/generate_tables.py`.
- [ ] `evaluation/comprehensive_uncertainty_eval.py`: shape mismatch (10776,) vs
  (21552,) — predictions appended twice for Bayesian models.
- [ ] Re-generate advection and uncertainty plots with current checkpoints.

## 4. Validation (ablation zone, CEM43)

- [ ] Run `make validation MODEL=<name> CKPT=<path>` with a trained dense model
  (registry loading + MC sampling exist; only the grayscale `stub` baseline is
  the default) and commit the resulting IoU/Dice.
- [ ] CEM43 threshold sweep (30–240 min) with IoU/Dice curves.
- [ ] Document GT capture protocol; replace intensity thresholding segmentation.
- [ ] Probabilistic ablation zone from UQ models.

## 5. Dense maps & LTC (Part 2/3)

- [ ] Finalize `evaluation/evaluate_dense.py` for dense-map metrics.
- [ ] Bayesian U-Net: variational bottleneck in `models/dense_heads.py`.
- [ ] Bayesian LTC: variational encoder for `LatentLTC_UNet`.
- [ ] ConvLTC LOSO MAE is 97 K — check before reporting.

## 6. Paper

- [ ] Measure edge latency on real hardware or label all FPS numbers as simulated.

## 7. XAI (merged from `feature/xai-integration`, low priority)

- [ ] `RegressionWrapper` only reduces dense outputs; scalar sequence models
  (e.g. SimpleResNet, output `(B, T)`) need a Captum `target` index.
- [ ] `FaithfulnessCorrelation` OOM in `evaluation/evaluate_xai.py`.
- [ ] `demo/xai_dashboard.py` flickers.
- [ ] Compare attributions of masked vs. unmasked models ("cheating" on the
  thermometer artifact, issue #46).

## Branches

Only `main` exists. On 2026-11-02 all fully merged feature branches were
deleted, `feature/xai-integration` was merged, and `writing/intro-and-setup` was
dropped because its only commit was already superseded on `main`.

## Done (recent)

- [x] Fold-level paper figures/tables (`make paper_figures`): per-fold MAE
  distribution, paired regularization plot, Holm-adjusted pairwise Wilcoxon
  heatmap + vs-best table, fold difficulty, robust stats, per-fold table; paper
  text + limitations updated and `paper/main.pdf` rebuilt.
- [x] `pairwise_stat_tests.py` pairs folds per model pair (an incomplete run no
  longer drops folds for every model); `results/pairwise_wilcoxon.csv` regenerated.

- [x] LOSO for 12 models × 15 folds (masked), aggregated in `results/loso_summary.csv`.
- [x] `--weight_decay`, `--augment`, `--seed` + per-fold provenance JSON in the LOSO driver.
- [x] Regularized LOSO for SimpleResNet, CNNLSTM, BayesianResNet, KANResNet;
  `paper/tables/loso_regularized.tex` filled with real numbers and Wilcoxon p-values.
- [x] `aggregate_loso.py` no longer pools run variants of the same architecture.
- [x] Thermometer artifact masking; data migration to `data/level1_cropped`.
- [x] Pre-commit, CI workflow, `pytest.ini`, global seeding.
- [x] Lint (2026-11-02): codebase formatted with Black/isort (commit listed in
  `.git-blame-ignore-revs`), flake8 findings fixed, lint settings in
  `pyproject.toml` + `.flake8` only; CI `lint` job green.
- [x] Fixed legacy uncertainty evaluation (2026-11-02): infer ensemble input and
  output dimensions from each checkpoint; scalar-head checkpoints compare to
  the mean sensor target, and incompatible ensemble members fail explicitly.
- [x] Spatial field visualization now pairs predictions with sparse sensor
  ground truth; `training/train.py` uses regression models with output shapes
  matching the final sequence labels.
- [x] ONNX export (2026-11-02): `utils/export_to_onnx.py` unwraps
  `{"model_state_dict": ...}` checkpoints and infers input channels (ResNetUNet is
  RGB); `tests/test_export_to_onnx.py` checks PyTorch vs. onnxruntime outputs.
- [x] uv environment (2026-11-02): `pyproject.toml` + committed `uv.lock`,
  `.venv` rebuilt with `uv sync --locked`, `requirements.txt` generated from the
  lock, CI on uv (see [decision 0003](docs/decisions/0003-uv-environment.md)).
- [x] Repository cleanup (2026-11-02): stale docs → `docs/archive/`, one-off
  scripts → `scripts/archive/`, `Makefile.loso` folded into `Makefile`, hardcoded
  `/mnt/data2/...` paths replaced, XAI branch merged.
