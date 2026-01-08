# Benchmark Status Report

**Date:** January 8, 2026
**Status:** Part 1 Retraining & Part 2/3 Benchmark Execution

## Part 1 Summary (Vector Regression)
Evaluation of single-value regression models is complete.
*   **Best Model:** Simple ResNet (RMSE 1.86°C).
*   **Physics Result:** Spatial Bioheat (RMSE 1.88°C) validated the PDE approach.
*   **Active Tasks:**
    *   `physics_cnnlstm`: Retraining in progress (Epoch 3/50, Loss ~724).
    *   `pretrained_cnnlstm`: Retraining in progress (Epoch 21/50, Loss ~650).

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

3.  **Physics Prior (Optional):**
    *   Implemented a Gaussian Heatmap Prior based on input wattage for Residual Learning ($Pred = Prior + Delta$).
    *   **Refinement:** Made this strictly optional (`--no_physics_prior`) to ensure the model can be trained purely on data if metadata is missing or assumptions fail.

4.  **Testing Infrastructure:**
    *   Established `tests/` directory with `pytest` suite.
    *   Covered: Dataset loading (shapes/metadata), Model Architecture (I/O shapes), and Physics Logic.

5.  **Hybrid Physics Training:**
    *   Implemented `BioheatHybridLoss` in `physics/hybrid_loss.py` (MSE + Bioheat PDE Residual).
    *   Created `training/train_unet_hybrid.py` to train with PDE constraints.

### Benchmarks (Active)
The following benchmarks are currently running in `tmux` (session `video_regression_benchmarks`):
*   `unet_sparse_noprior`: U-Net Baseline (running).
*   `unet_sparse_withprior`: U-Net + Gaussian Prior (running).
*   `unet_hybrid_physics`: U-Net + Bioheat PDE Reg (running).

### Next Steps (Implementation Plan)
1.  **Evaluate Results:** Compare convergence and final MSE once training completes.
2.  **Part 3 (Time):**
    *   **Documentation:** Added a new section to the paper (`ltc_section.tex`) justifying the use of Liquid Time Constant (LTC) networks via the Bioheat Transfer Equation.
    *   **Implementation:** Implemented `models/latent_ltc.py` (Latent-Space Dynamics) and `training/train_ltc.py`.
    *   **Status:** Active. `ltc_unet_seq16_hybrid` and `conv_ltc_seq16_hybrid` are training.

3.  **Verification:**
    *   **Test Suite:** Expanded `tests/test_model_suite.py` covering all Part 2 and Part 3 architectures (`ResNetUNet`, `LatentLTC_UNet`, `CNNLSTM`, `SimpleResNet`).
    *   **Status:** All tests passing.
