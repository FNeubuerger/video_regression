# Benchmark Status Report

**Date:** January 14, 2026
**Overall Status:** LOSO Cross-Validation initiated; Inverse Physics (Spatial Params) in training.

## Project Performance Overview

The following table summarizes the performance across all major model variants tested in the repository.

| Part | Category | Model Name | Test RMSE (K) | Test MAE (K) | Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **P2** | **Inverse Physics** | `SpatialPhysicsCNNLSTM` | -- | -- | **Training (GPU 1)** |
| **P2** | **Physics** | `ConvectionBioheat` | **0.62** | **0.24** | **Unmasked Leader** |
| **P2** | **Physics** | `MetabolicBioheat` | **0.84** | **0.21** | **Unmasked Leader** |
| **P3** | **Uncertainty** | `BayesianCNNLSTM` | 1.85 | 0.83 | **Evaluating** |
| **P4** | **Validation** | `LOSO (Baseline)` | -- | -- | **Running (GPU 0)** |
| **P4** | **Validation** | `LOSO (Physics)` | -- | -- | **Running (GPU 1)** |
| **P4** | **Validation** | `LOSO (Spatial Physics)`| -- | -- | **Running (GPU 1)** |

---

## Detailed Component Status

### 1. LOSO Cross-Validation (Active)
*   **Update:** Started Leave-One-Sequence-Out (LOSO) benchmarks to test spatial normalization performance.
*   **Infrastructure:** parallel tmux sessions `loso_baseline` and `loso_physics`.

### 2. Inverse Physics & Spatial Parametrization (Issue #41, #42)
*   **Progress:** Implemented `SpatialPhysicsCNNLSTM` which predicts learnable perfusion ($\alpha$) and conductivity ($\beta$) maps.
*   **Advection:** Integrated Dense Optical Flow into the physics loss term.
*   **Visualization:** New tool [evaluation/visualize_advection.py](evaluation/visualize_advection.py) ready for advection analysis.


## Next High-Level Actions
1.  **Monitor Retraining:** Observe `masked_retraining` and `bayesian_pinn_masked` sessions.
2.  **Scientific Plots:** Extend [evaluation/generate_scientific_plots.py](evaluation/generate_scientific_plots.py) to compare Masked vs Unmasked profiles once training finishes.
3.  **XAI Validation:** Run GradCAM on the new masked checkpoints (Issue #46).
