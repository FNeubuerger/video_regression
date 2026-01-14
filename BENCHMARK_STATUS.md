# Benchmark Status Report

**Date:** January 14, 2026
**Overall Status:** Phase 2 (Physics) Validating; Phase 3 (Uncertainty) - Masked Retraining Initiated.

## Project Performance Overview

The following table summarizes the performance across all major model variants tested in the repository.

| Part | Category | Model Name | Val RMSE (°C) | Val MAE (°C) | Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **P1** | **Baselines** | `SimpleResNet` | 2.97 | 1.31 | **Completed** |
| | | `CNNLSTM` | 44.46 | 32.27 | **Completed** |
| **P2** | **Physics** | `BioheatPINN` (BTE) | **1.15** | **0.26** | **Validating (Masked)** |
| | | `ConvectionBioheat` | **0.80** | **0.29** | **Validating (Masked)** |
| | | `SpatialMetabolic` | 1.83 | 0.72 | **Validating (Masked)** |
| **P3** | **Uncertainty** | `BayesianCNNLSTM` | 34.50* | 28.50* | **Restarting (Masked)** |
| | | `ConvLTC_Hybrid` | 3.9e+6* | - | **Evaluating** |

---

## Detailed Component Status

### 1. Baselines & Standard Regression (Completed)
*   **Leader:** `SimpleResNet` (MAE: 1.31 K).
*   **Update:** Masking logic (Hough Transform for antennas) fully implemented. Baseline models remain relevant for comparison but unmasked versions likely "cheat" on artifact highlights.
*   **Implementation:** Automated antenna detection using Hough Transform and thermometer location masking via `sensor_coordinates.json` is now integrated into the dataset loader.

### 2. Physics-Informed Variants (In Validation)
*   **Leader:** `BioheatPINN` (MAE: **0.26 K**).
*   **Update:** `AdvancedBioHeatLoss` now supports spatial masking. Retraining triggered to ensure the physics consistency holds in tissue regions only.
*   **Stability:** Added `smoothness_weight` and `monotonicity_weight` to the physics loss functions to prevent unphysical oscillations during training.

### 3. Bayesian & Uncertainty Estimation (Relaunching)
*   **Status:** Previous Bayesian runs showed high instability (MAE > 30K).
*   **Update:** Automated masking for thermometers and antennas is now active. Masked retraining is mandatory for Phase 3 models.
*   **Progress:** `ConvLTC_Hybrid` finished training (Jan 14). Evaluation in progress.


## Next High-Level Actions
1.  **Unified Evaluation:** Run [evaluation/run_unified_evaluation.py](evaluation/run_unified_evaluation.py) in tmux session `evaluation_session`.
2.  **Masked Retraining:** Relaunch [training/train_all_models.py](training/train_all_models.py) with `--masked` flag in tmux session `masked_retraining`.
3.  **LTC Evaluation:** Benchmarking the finished `ConvLTC_Hybrid` model for temporal stability.
4.  **XAI Validation:** Compare masked vs unmasked attributions (Issue #46).
