# Benchmark Status Report

**Date:** January 9, 2026
**Overall Status:** Phase 2 (Physics) Completed; Phase 3 (Uncertainty) in Progress.

## Project Performance Overview

The following table summarizes the performance across all major model variants tested in the repository.

| Part | Category | Model Name | Val RMSE (°C) | Val MAE (°C) | Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **P1** | **Baselines** | `SimpleResNet` | 2.97 | 1.31 | **Completed** |
| | | `CNNLSTM` | 44.46 | 32.27 | **Completed** |
| **P2** | **Physics** | `BioheatPINN` (BTE) | **1.15** | **0.26** | **Completed** |
| | | `ConvectionBioheat` | **0.80** | **0.29** | **Completed** |
| | | `SpatialMetabolic` | 1.83 | 0.72 | **Completed** |
| **P3** | **Uncertainty** | `BayesianCNNLSTM` | 34.50* | 28.50* | *Training* |

---

## Detailed Component Status

### 1. Baselines & Standard Regression (Completed)
*   **Leader:** `SimpleResNet` (MAE: 1.31 K).
*   **Observation:** Performance dropped slightly on the full test set compared to initial subsets.
*   **Update:** Masking logic implemented (Jan 9). Retraining triggered.

### 2. Physics-Informed Variants (Completed)
*   **Leader:** `BioheatPINN` (MAE: **0.26 K**).
*   **Benefit:** Incorporating the Bioheat Transfer Equation (BTE) reduced error by over 80%.
*   **Update:** Verification of "real learning" vs "cheating" is in progress using masked training.

### 3. Bayesian & Uncertainty Estimation (Active)
*   **Current Progress:** Evaluating Calibration (PICP) and Sharpness (MPIW).
*   **Status:** `BayesianCNNLSTM` (MAE 30.1 K) and `BayesianResNet` (MAE 29.6 K) show very high error. They appear to struggle with the raw sequence noise or lack of spatial grounding without masking.
*   **Update:** Masked retraining is critical to see if these models stabilize.


## Next High-Level Actions
1.  **Masked Retraining:** Relaunch [training/train_all_models.py](training/train_all_models.py) with `--masked` flag to establish clean benchmarks.
2.  **Uncertainty Evaluation:** Run [evaluation/comprehensive_uncertainty_eval.py](evaluation/comprehensive_uncertainty_eval.py) on finished Bayesian models.
3.  **XAI Validation:** Compare masked vs unmasked attributions (Issue #46).
4.  **Paper Prep:** Update results tables in the LaTeX manuscript with the latest metrics.
