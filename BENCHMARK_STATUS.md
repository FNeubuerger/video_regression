# Benchmark Status Report

**Date:** January 8, 2026
**Status:** Evaluation / Recovery Phase

## Executive Summary
Evaluation of the **Spatial Models** and baseline **Simple ResNet** is complete.
The **Simple ResNet** remains the performant baseline (RMSE 1.86°C), but the **Spatial Bioheat** model is extremely close (RMSE 1.88°C), validating the physics-informed approach in the spatial domain.

## Completed Benchmarks (Evaluation Results)

| Model | RMSE (°C) | MAE (°C) | R² Score | Status |
|-------|-----------|----------|----------|--------|
| **Simple ResNet** | **1.856** | **0.575** | **0.996** | **SOTA** |
| **Spatial Bioheat** | 1.882 | 1.188 | 0.996 | Competitive |
| **Spatial Metabolic** | 2.030 | 1.089 | 0.996 | Strong |
| **Spatial Convection** | 3.437 | 2.196 | 0.988 | Underperforming |
| **CNNLSTM** | 26.345 | 17.888 | 0.268 | **Needs Review** |

*Note: The CNNLSTM model performance is unexpectedly low compared to previous runs, likely due to a checkpoint issue or data loading mismatch during the new parallel evaluation. This requires investigation.*

## Failed / In Progress

| Model | Status | Cause/Notes | Action Items |
|-------|--------|-------------|--------------|
| **Physics CNNLSTM** | **Failed** | `python: not found` in Makefile. | **Fixed & Ready to Retrain.** |
| **Pretrained CNNLSTM** | **Failed** | `python: not found` in Makefile. | **Fixed & Ready to Retrain.** |
| **Ensemble** | **Stalled** | Stopped at Member 4/5. | Investigate logs/resources. **Restart Member 4 & 5.** |

## Technical Incident Report

### 1. Makefile Environment Error (Resolved)
*   **Issue:** `Physics CNNLSTM` and `Pretrained CNNLSTM` logs show `/bin/sh: 1: python: not found`.
*   **Fix:** Updated `Makefile` to use `python3`.

### 2. Evaluation Optimization (Resolved)
*   **Action:** Optimized `evaluate_models.py` to run inference for **all models simultaneously** using `DataParallel` on the A100 GPUs.
*   **Result:** Reduced evaluation time ~5x (Single pass over test set for 5 models).

## Next Steps
1.  **Retrain Failed Models:** Launch Physics/Pretrained benchmarks using the fixed Makefile.
2.  **Restart Ensemble:** Resume the stalled ensemble training.
3.  **Investigate CNNLSTM:** Check why the baseline CNNLSTM evaluated poorly (possible normalization or sequence length mismatch in evaluation script).
