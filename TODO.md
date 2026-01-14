# TODO: Project Status & Issues

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
- [ ] **Rerun Benchmarks:** Execute full training/evaluation pipeline for all models using LOSO to prove setup generalization.
    - [ ] Compare LOSO results (Generalization) vs. Random Split results (Memorization).
    - [ ] Maintain legacy benchmark results for internal comparison.

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
