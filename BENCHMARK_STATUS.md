# Benchmark Status Report

**Date:** January 14, 2026
**Overall Status:** **CRITICAL UPDATE: RE-STARTING ALL BENCHMARKS ON NEW DATASET (level1_cropped).** Prior results were on legacy data. Full retraining initiated.

## Project Performance Overview

The following table summarizes the performance across all major model variants tested in the repository.

| Part | Category | Model Name | Test RMSE (K) | Test MAE (K) | Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **P2** | **Inverse Physics** | `SpatialPhysicsCNNLSTM` | -- | -- | **Pending Retrain** |
| **P2** | **Physics** | `ConvectionBioheat` | -- | -- | **Pending Retrain** |
| **P3** | **Uncertainty** | `BayesianCNNLSTM` | -- | -- | **Pending Retrain** |
| **P4** | **Benchmark** | `Baseline CNNLSTM` | -- | -- | **Pending Retrain** |
| **P4** | **Benchmark** | `Physics Informed` | -- | -- | **Pending Retrain** |
| **P4** | **Benchmark** | `Bayesian ResNet` | -- | -- | **Pending Retrain** |

*> **NOTE:** Previous results have been archived. All models are being retrained on the verified `level1_cropped` dataset to ensure consistency with the new paper methodology.*

---

## Detailed Component Status

### 1. Dataset Migration (Completed)
*   **Action:** Verified that `training/train_all_models.py` was pointing to legacy data.
*   **Resolution:** Updated all training and evaluation scripts to use `SequenceHeatmapDataset` pointing to `data/level1_cropped`.
*   **Verification:** Visualized ROI context and advection fields to confirm alignment.

### 2. Full Benchmark Retraining (Active)
*   **Scope:** Training 8 major architectures from scratch on the new dataset.
*   **Old Models:** Archived to `models/archive_legacy_data`.
*   **Strategy:** Retraining Unmasked first, followed by Masked variants.

### 3. Evaluaton Pipeline
*   **Aggregation:** [evaluation/generate_tables.py](evaluation/generate_tables.py) updated to consolidate LOSO results (Mean $\pm$ Std across sequences).
*   **Visualization:** New tool [evaluation/visualize_loso.py](evaluation/visualize_loso.py) added to generate boxplots and error heatmaps per sequence.
*   **Units:** All outputs standardized to SI derived units ($T/K$).


## Next High-Level Actions
1.  **Execute Training:** Run `train_all_models.py` for standard and masked models.
2.  **Monitor Convergence:** Ensure loss curves on new data look healthy.
3.  **Scientific Plots:** Extend [evaluation/generate_scientific_plots.py](evaluation/generate_scientific_plots.py) to compare Masked vs Unmasked profiles once training finishes.

