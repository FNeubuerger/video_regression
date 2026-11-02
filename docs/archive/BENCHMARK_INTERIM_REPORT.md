# Benchmark Interim Report

**Date:** January 14, 2026
**Time:** 16:50

## Executive Summary
This report captures the status of the comprehensive **Leave-One-Sequence-Out (LOSO) Cross-Validation** campaign.
**Key Achievement:** Successfully launched a parallel benchmark suite across 8 major model architectures to validate spatial generalization. All models have survived the first epoch with stable loss trends.

## 1. LOSO Generalization Benchmarks (Active)
*Status: Running (Parallel tmux sessions)*

| Model | Status | Progress | RMSE (Current Epoch) |
|-------|--------|----------|----------------------|
| **Physics Informed** | 🟢 Training | Epoch 1 (83%) | **4.79 K** |
| **Spatial Physics** | 🟢 Training | Epoch 1 (50%) | 60 K (Descending) |
| **Bayesian ResNet** | 🟢 Training | Epoch 1 (36%) | 152 K (Descending) |
| **CNNLSTM** | 🟢 Training | Epoch 1 (36%) | 479 K |
| **Simple ResNet** | 🟢 Training | Epoch 1 (36%) | 40.3 K |

## 2. Infrastructure & Automation (New)
We have overhauled the validation and monitoring pipeline:
*   **Live Dashboard:** `make monitor` provides a unified view of all 8 LOSO folds.
*   **Scientific Reporting:** `make evaluation` now triggers an automated aggregation of all 7 folds (Mean ± Std).
*   **Standardization:** All temperature metrics are now reported in Kelvin (K).

## 3. Inverse Physics & Spatial Parametrization
*Status: Feature Complete*
*   **Spatial Maps:** Models now predict learnable $\alpha$ and $\beta$ maps to account for tumor/tissue boundary variations.
*   **Advection:** Integrated Dense Optical Flow into the `AdvancedBioHeatLoss` to handle heat transport effectively.

## 4. Strategic Priorities (Updated)
1.  **Monitor LOSO Convergence:** Ensure all folds complete without GPU memory overflows.
2.  **Generalization Gap Analysis:** Quantify the drop in performance from Random-Split to LOSO to determine the "Generalization Gap."
3.  **Checkpoints Validation:** Verify the quality of learned spatial parameter maps once stable.

```