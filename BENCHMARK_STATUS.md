# Benchmark Status Report

**Date:** January 15, 2026
**Overall Status:** **RETRAINING IN PROGRESS.** Initial results for Standard Models available. Uncertainty and Spatial Physics models encountered errors and are being patched.

## Project Performance Overview

The following table summarizes the performance across all major model variants tested in the repository.

| Category | Model Name | Validation Loss (MSE) | Epochs | Status |
| :--- | :--- | :---: | :---: | :--- |
| **Standard** | `CNNLSTM` | **1.1842** | 50 | ✅ Completed |
| **Standard** | `PretrainedCNNLSTM` | **0.4609** | 32 | ✅ Completed (Early Stop) |
| **Standard** | `SimpleResNet` | **0.4723** | 47 | ✅ Completed |
| **Physics** | `PhysicsCNNLSTM` | **2.4260** | 43 | ✅ Completed |
| **Physics** | `SpatialPhysicsCNNLSTM` | -- | -- | ❌ Failed (Unpacking Error) |
| **Uncertainty** | `Ensemble` | -- | -- | ⚠️ Interrupted (Restarting) |

*> **NOTE:** Results are based on the new `level1_cropped` dataset. Validation Loss is Mean Squared Error on normalized targets.*

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

