# TODO: Project Status & Issues

## Evaluation Pipeline Repairs (Critical)
- [ ] **Fix Shape Mismatches in `evaluate_models.py`**
    - **Error**: `ValueError: operands could not be broadcast together with shapes (10776,) (43104,)`.
    - **Analysis**: Mismatch between prediction count (43104) and ground truth count (10776). Ratio is ~4. Likely related to sequence length (5) minus 1, or some frame-level vs sequence-level output discrepancy in legacy models (`PretrainedCNNLSTM`, `SimpleResNet`).
    - **Action**: Debug `evaluate_model` method to correctly reduce frame-level predictions to sequence-level (e.g., mean, or taking the last frame), or filter out legacy models.
- [ ] **Fix Shape Mismatches in `comprehensive_uncertainty_eval.py`**
    - **Error**: `ValueError: operands could not be broadcast together with shapes (10776,) (21552,)`.
    - **Analysis**: Ratio is exactly 2. Likely related to `BayesianResNet` or `BayesianCNNLSTM` returning duplicate predictions per sample, or an accumulation error in the evaluation loop (appending twice?).
    - **Action**: Inspect the `evaluate_model` loop in `comprehensive_uncertainty_eval.py` for double appending or incorrect concatenation of means/stds.
- [ ] **Restore `metrics_comparison.csv` generation**
    - **Issue**: File is not created because `evaluate_models.py` crashes before the saving step.
    - **Impact**: `paper/viz_performance.py` and `generate_tables.py` fail.
- [ ] **Cleanup Legacy Models**: User indicated legacy models don't need to work. Consider removing them from the automated evaluation queue to stabilize the pipeline for NEW models.


## Critical Issues
- [x] **Data Migration**: Models were training on legacy `data/` instead of `data/level1_cropped`.
    - **Fix**: Updated `SequenceHeatmapDataset` and all training scripts to point to `level1_cropped`.
    - **Status**: Verified dataset logic and visual alignment.
- [ ] **Benchmark Integrity**: Re-run all benchmarks on the new dataset.
    - **Status**: Old results archived. Full retraining initiated (Unmasked & Masked).
    - **Action**: Monitor training sessions.

## Upcoming Tasks
- [ ] **Transfer Learning Investigation**: Investigate using archived legacy models as seed weights for the new dataset.
- [ ] **Scientific Plots**: Re-generate advection and uncertainty plots once new checkpoints are available.

## Resolved
### Fix Spatial Model Crashes
- **Issue**: `RuntimeError` due to dimension mismatch in `AdvancedBioHeatLoss`.
- **Fix**: Added dummy temporal dimension in validation loops.

- [x] Fix validation loop in `training/train_spatial_bioheat.py`
- [x] Fix validation loop in `training/train_spatial_convection_bioheat.py`
- [x] Fix validation loop in `training/train_spatial_metabolic_bioheat.py`
- [ ] Rerun the benchmarks to verify the fix.

## Priority Actions (Jan 9)
- [x] **Core Evaluation:** Run GPU-accelerated evaluation for all deterministic models.
    - Achieved MAE 0.26K with `BioheatPINN`.
    - Integrated LaTeX table generation.
- [x] **Thermometer Masking (#41):** Implement spatial artifact masking in `heatmap_dataset.py`.
    - Avoid models "cheating" by looking at bright sensor spots.
    - Added `use_artifact_masking` to `TemperatureSequenceDataset`.
- [x] **Bayesian Evaluation:** Run `evaluation/comprehensive_uncertainty_eval.py` for all Tier 3 models.
    - `BayesianCNNLSTM` summary done. (MAE ~30)
    - `BayesianResNet` done. (MAE ~29.6)
    - Created `evaluation/run_unified_evaluation.py` and `make evaluation` target.
- [x] **XAI Comparative Analysis:** Created Issue #46 to track cheating vs. learning analysis.
- [ ] **Retraining:** Launch masked training benchmark via `training/train_all_models.py --masked`.

## LOSO Cross-Validation (#47)
- [x] **Implementation:** `evaluation/loso_cross_validation.py` created with fold-based logic.
- [x] **Rerun Benchmarks:** Execute full training/evaluation pipeline for all models using LOSO to prove setup generalization.
    - [x] Compare LOSO results (Generalization) vs. Random Split results (Memorization). See `paper/tables/loso_summary.tex` and `paper/main.tex` ("Cross-Subject Generalization").
    - [x] Maintain legacy benchmark results for internal comparison.

## LOSO Overfitting Mitigation (#48)
- **Issue:** LOSO folds trained with unregularized AdamW (no weight decay), no augmentation, and non-deterministic seeding, producing a large mixed-split vs. LOSO generalization gap (e.g. SimpleResNet 1.31 K -> 26.67 K MAE).
- [x] Added `--weight_decay` (default `1e-4`), `--augment` (flip + brightness jitter via `utils/augmentation.py`), and `--seed` (via `utils/seed_utils.py`) to `evaluation/loso_cross_validation.py`.
- [x] Per-fold provenance export: `checkpoints/loso/<model>/fold_<id>_config.json` (seed, weight decay, git sha, hardware, timestamp).
- [x] `scripts/run_loso_benchmarks_regularized.sh` launches regularized retraining for `SimpleResNet` and `CNNLSTM` (the two worst-affected models) across 2 GPUs via tmux.
- [ ] **In progress:** retraining running in tmux sessions `loso_reg_simpleresnet` / `loso_reg_cnnlstm` (15 folds x 10 epochs each). Once complete, fill in `paper/tables/loso_regularized.tex` and extend the regularized run to remaining overfitting-prone models (`ConvectionBioheat`, `BayesianResNet`, `KANResNet`).

## Part 2/3 (Dense & LTC) Analysis
- [x] **Evaluation:** Physics-Informed models (`BioheatPINN`) significantly outperform baselines.
- [x] **Cleanup:** Organized project structure (moved scripts to `scripts/`, `utils/`).
- [ ] **Implementation:** Finalize `evaluation/evaluate_dense.py` for dense map metrics.
- [ ] **Bayesian U-Net:** Implement "Probabilistic Bottleneck" (Variational Layer) in `models/dense_heads.py`.
- [ ] **Bayesian LTC:** Implement Variational Encoder for `LatentLTC_UNet`.

## XAI & Interpretability (Bonus / Deprioritized)
- [x] **Justification:** Created `research/XAI_PAPER_CONTRIBUTION.md` (on `feature/xai-integration`).
- [x] **Metric Implementation:** `evaluation/evaluate_xai.py` implemented (on `feature/xai-integration`).
- [ ] **Fix Memory Error:** `FaithfulnessCorrelation` causing OOM. (Low Priority)
- [ ] **Dashboard Fix:** `demo/xai_dashboard.py` flickers. (Low Priority)


## XAI & Interpretability (Bonus / Deprioritized)
- [x] **Justification:** Created research/XAI_PAPER_CONTRIBUTION.md (on xai branch).
- [x] **Metric Implementation:** evaluation/evaluate_xai.py implemented (on xai branch).
- [ ] **Fix Memory Error:** FaithfulnessCorrelation causing OOM. (Low Priority)
- [ ] **Dashboard Fix:** demo/xai_dashboard.py flickers. (Low Priority)
