# Benchmark Interim Report

**Date:** January 8, 2026
**Time:** 16:45

## Executive Summary
This report captures the status of ongoing benchmarks across both active branches.
**Key Finding:** The **ConvLTC** model (Part 3) is achieving the best data fit (MSE ~288) among complex temporal models, significantly outperforming the standard U-Net Hybrid (MSE ~780), though it incurs a high physics penalty.

## 1. Advanced Physics Models (Branch: `feature/advanced-physics`)
*Status: Healthy / Recovering*

| Model | Status | Progress | Notes |
|-------|--------|----------|-------|
| **Physics CNNLSTM** | 🟢 Training | Epoch 12/50 | Recovered from crash. Loss decreasing steadily. |
| **Advanced Bioheat CNNLSTM** | 🟢 Training | Epoch 33/50 | Stable. |
| **Bayesian CNNLSTM** | 🟢 Training | Epoch 32/50 | Stable. |
| **Convection/Metabolic** | 🟢 Training | Epoch 33/50 | Stable. |

## 2. Dense Map & Temporal Dynamics (Branch: `feature/dense-map-estimation`)
*Status: Active (tmux session: `video_regression_benchmarks`)*

These models predict full $H \times W$ temperature maps, incorporating PDE constraints.

| Model | Epoch | Train Loss | Val Loss | MSE (Data Fit) | Physics Loss | Analysis |
|-------|-------|------------|----------|----------------|--------------|----------|
| **U-Net Hybrid** | 10/50 | 1,227 | 1,437 | ~780 | ~447,000 | **Balanced.** Learning both data and physics. Physics loss dropped ~40% in 5 epochs. |
| **Latent LTC** | 14/50 | 129,000 | 220,000 | ~680 | High | **Slow Convergence.** Stable but struggling to reduce the scale of the loss. |
| **ConvLTC** | 17/50 | 4.2M | 4.2M | **~288** | 4,200,000 | **Best Data Fit.** Excellent MSE, but effectively "ignoring" the physics constraint (high penalty). |

## 3. Strategic Priorities (Updated)
Based on recent review, the priorities for the immediate future are:

1.  **Core Evaluation (Part 2 & 3):**
    *   Wait for `ConvLTC` and `U-Net Hybrid` to finish.
    *   Perform rigorous quantitative comparison (RMSE, SSIM, Physics Compliance).
    *   Generate comparative plots for the paper.
2.  **Paper Finalization:**
    *   Fill in the Results section with these new benchmark numbers.
    *   Finalize the Methodology section for ConvLTC.
3.  **XAI (Bonus):**
    *   Deprioritized. Will only pursue single-frame static explanations if time permits after core results are solidified.
