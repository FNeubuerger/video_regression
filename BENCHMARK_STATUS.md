# Benchmark Status Report

**Date:** January 14, 2026
**Overall Status:** LOSO Cross-Validation initiated for all models; Monitoring pipeline active.

## Project Performance Overview

The following table summarizes the performance across all major model variants tested in the repository.

| Part | Category | Model Name | Test RMSE (K) | Test MAE (K) | Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **P2** | **Inverse Physics** | `SpatialPhysicsCNNLSTM` | -- | -- | **Training (Fold 1)** |
| **P2** | **Physics** | `ConvectionBioheat` | **0.62** | **0.24** | **Unmasked Leader** |
| **P3** | **Uncertainty** | `BayesianCNNLSTM` | 1.85 | 0.83 | **Evaluating** |
| **P4** | **LOSO** | `Baseline CNNLSTM` | 2.45e+3* | -- | **In Progress (Epoch 1)** |
| **P4** | **LOSO** | `Physics Informed` | 29.8 | -- | **In Progress (Epoch 1)** |
| **P4** | **LOSO** | `Bayesian ResNet` | 83.7 | -- | **In Progress (Epoch 1)** |

*\*High initial loss expected in first epoch.*

---

## Detailed Component Status

### 1. LOSO Cross-Validation (Active)
*   **Update:** Started comprehensive LOSO benchmarks for all 8 major architectures.
*   **Infrastructure:** parallel tmux sessions managed by `Makefile`.
*   **Monitoring:** New shell script [scripts/monitor_loso.sh](scripts/monitor_loso.sh) used to track progress across all concurrent sessions.

### 2. Evaluaton Pipeline
*   **Aggregation:** [evaluation/generate_tables.py](evaluation/generate_tables.py) updated to consolidate LOSO results (Mean $\pm$ Std across sequences).
*   **Visualization:** New tool [evaluation/visualize_loso.py](evaluation/visualize_loso.py) added to generate boxplots and error heatmaps per sequence.
*   **Units:** All outputs standardized to SI derived units ($T/K$).


## Next High-Level Actions
1.  **Monitor Retraining:** Observe `masked_retraining` and `bayesian_pinn_masked` sessions.
2.  **Scientific Plots:** Extend [evaluation/generate_scientific_plots.py](evaluation/generate_scientific_plots.py) to compare Masked vs Unmasked profiles once training finishes.
3.  **XAI Validation:** Run GradCAM on the new masked checkpoints (Issue #46).
