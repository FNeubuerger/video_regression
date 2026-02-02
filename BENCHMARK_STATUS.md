# Benchmark Status Report

**Date:** January 19, 2026
**Overall Status:** **ACTIVE.**
1.  **Retraining Failed Streams:** Bayesian and Physics models crashed due to dataset issues are retraining.
2.  **LOSO Benchmark:** Freshly launched for all core models.
3.  **Killed Streams:** Legacy streams (Ensemble, U-Net) were interrupted by server restart.

## 1. Active Training Streams

| Stream | Component | Models Included | Status | Log Location | Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Retrain** | `restart_failed_due_to_dataset.sh` | Bayesian, Convection | 🚀 **Running** (Restarted) | `logs/retrain/` | Standard Train (80/20) |
| **2. LOSO** | `scripts/run_loso_benchmarks.sh` | All Core Models (x15) | 🚀 **Running** (New) | `logs/loso/` | Cross Validation |

### LOSO Details (Launched 2026-01-19)
Running Leave-One-Sequence-Out CV for:
*   `CNNLSTM`, `PretrainedCNNLSTM`
*   `PhysicsCNNLSTM`, `ConvectionBioheat`
*   `SimpleResNet`, `SpatialResNet`
*   `BayesianResNet`, `FullBayesianResNet`, `BayesianCNNLSTM`
*   `ConvLTC`

---

## 2. Completed Benchmarks (Phase 1 & 2 - Standard Split)

| Category | Model Name | Val Loss (MSE) | Epochs | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Standard** | `CNNLSTM` | **13.6705** | 23 | Early stopping. Poor performance compared to Pretrained. |
| **Standard** | `PretrainedCNNLSTM` | **617.58** | 24 | (Old Run) High loss. |
| **Standard** | `SimpleResNet` | **2.93** | 33 | (Old Run) Reasonable baseline. |
| **Physics** | `PhysicsCNNLSTM` | **2.4260** | 43 | High loss likely due to regularization. |
| **Physics** | `Bioheat PINN` (Scalar) | **0.1009** | 30 | Excellent performance. |
| **Physics** | `Metabolic Bioheat` (Scalar) | **877.30** | 30 | Extremely high loss. |
| **Physics** | `Spatial Bioheat` | **3.0952** | 30 | Reasonable spatial convergence. |
| **Physics** | `Spatial Metabolic` | **1240.89** | 30 | Failed to converge properly. |

---

## 3. Pending / Killed / Failed Benchmarks

### A. Killed (User Intervention / Server Restart)
*   **ConvLTC**: Terminated. No checkpoint found.
*   **LatentLTC**: Terminated. No checkpoint found.
*   **LatentLTC Variational**: Terminated. No checkpoint found.
*   **Ensemble**: Terminated at Epoch 4/20.
*   **U-Net Stream**: Terminated at Epoch 1.
*   **Spatial Convection Bioheat**: Terminated at Epoch 7/30. (Checkpoint exists).

### B. Restarting (Dataset Error Fix)
The following models caused a dataset collate error and are currently retraining in the **Retrain** session:
*   `Bayesian CNNLSTM`, `Bayesian PINN`
*   `Bayesian Convection PINN`, `Bayesian Metabolic PINN`
*   `Convection Bioheat` (Scalar)
*   `Bayesian Spatial Convection`

---

## 4. Evaluation & Next Steps

### Action Items
1.  **Monitor LOSO:** Ensure cross-validation sessions start correctly (`tmux ls`).
2.  **Monitor Retrain:** Watch for stability in the restarted physics models.
3.  **Evaluate Completed:** Run `make evaluation` or `python evaluation/run_unified_evaluation.py` for models that have finished.

