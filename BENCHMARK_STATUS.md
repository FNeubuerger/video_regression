# Benchmark Status Report

**Date:** January 8, 2026
**Status:** Part 2: Dense Map Estimation Implementation

## Part 1 Summary (Vector Regression)
Evaluation of single-value regression models is complete.
*   **Best Model:** Simple ResNet (RMSE 1.86°C).
*   **Physics Result:** Spatial Bioheat (RMSE 1.88°C) validated the PDE approach.
*   **Pending:** Retraining CNNLSTM variants (low priority compared to Part 2).

## Part 2 Status: Dense Map Estimation
We have transitioned the codebase to support dense temperature map prediction ($H \times W$ heatmap output instead of scalar).

### Completed Work
1.  **Refactored Data Pipeline (Level 0 $\to$ Level 1):**
    *   **Preprocessing:** Implemented dynamic "Active Zone" cropping ($Y_{peak} \pm 180$) to standardize inputs.
    *   **Sensor Localization:** Implemented robust Axis-Aligned 4-sensor detection on clean, averaged frames.
    *   **Dataset:** Created `TemperatureHeatmapDataset` that synchronizes video frames with CSV logs and generates sparse supervision masks.
    *   **Validation:** Generated verification videos confirming sensor alignment and coordinate stability.

2.  **Architecture:**
    *   Implemented `ResNetUNet` (Encoder-Decoder with Skip Connections) in `models/dense_heads.py`.

### Next Steps (Implementation Plan)
1.  **Hybrid Loss Function:** Implement `HybridLoss = MSE_Sparse + lambda * PDE_Dense`.
2.  **Training Loop:** Create training script for the U-Net.
3.  **Baseline Training:** Train a standard U-Net with Sparse MSE only (no physics).
4.  **Physics Training:** Train Physics-Informed U-Net.
