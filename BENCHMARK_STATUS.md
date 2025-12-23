# Benchmark Status Report

**Date:** December 23, 2025
**Status:** Active / Retraining Phase

## Executive Summary
We have successfully benchmarked the baseline models. **Simple ResNet** is currently the State-of-the-Art (SOTA) with a Validation Loss of **2.93**, significantly outperforming the basic **CNNLSTM** (Loss: 13.67).

The complex physics-informed models (`Physics CNNLSTM`, `Pretrained CNNLSTM`) initially failed to converge (Loss > 600). We diagnosed this as an **exploding gradient problem** in the LSTM layers. A fix (Gradient Clipping) has been implemented, and these models are currently being retrained.

## Completed Benchmarks

| Model | Best Validation Loss | Training Time | Notes |
|-------|----------------------|---------------|-------|
| **Simple ResNet** | **2.9318** | ~8.5h | **Current SOTA**. Demonstrates that single-frame regression is highly effective. |
| **Bayesian Head** | 11.0647 | - | Good uncertainty quantification with minimal accuracy trade-off. |
| **CNNLSTM** | 13.6705 | ~6h | Decent performance, but lags behind the simpler ResNet. |
| **Full Bayesian** | 15.0593 | - | Higher loss expected due to KL-divergence regularization term. |
| **Physics CNNLSTM** | - | - | **Retraining** (Gradient Clipping Fix) |
| **Pretrained CNNLSTM** | - | - | **Retraining** (Gradient Clipping Fix) |

## In Progress

| Model | Status | Notes |
|-------|--------|-------|
| **Ensemble** | Training Member 4/5 | Robust uncertainty estimation. Currently at Epoch 16/20 for member 4. |
| **Spatial Bioheat** | Epoch 3/50 | Restarted after validation fix. Training stable. |
| **Spatial Convection** | Epoch 3/50 | Restarted after validation fix. Training stable. |
| **Spatial Metabolic** | Epoch 3/50 | Restarted after validation fix. Training stable. |
| **Physics/Pretrained Retrain** | Started | Retraining with `max_norm=1.0` gradient clipping. |

## Technical Incident Report

### 1. Spatial Model Validation Crash (Resolved)
*   **Issue:** Shape mismatch in validation loop (`tensor a (4) must match tensor b (32)`).
*   **Cause:** Loss function expected temporal dimension `(B, T, ...)` but received `(B, ...)` in validation.
*   **Fix:** Added `.unsqueeze(1)` to predictions and labels in validation loops for all spatial models.
*   **Status:** Verified. Models are training successfully.

### 2. Physics Model Convergence Failure (Resolved)
*   **Issue:** `Physics CNNLSTM` and `Pretrained CNNLSTM` plateaued at high loss (~640 MSE).
*   **Diagnosis:** Debugging revealed **exploding gradients** in the LSTM layers (Gradient Norms > 800).
*   **Fix:** Implemented **Gradient Clipping** (`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`) in `training/train_all_models.py`.
*   **Verification:** Test run showed loss dropping from ~3000 to ~87 in <50 batches.
*   **Status:** Retraining initiated.

## Next Steps for Collaborators
1.  **Monitor Retraining:** Check `logs/physics_cnnlstm.log` and `logs/pretrained_cnnlstm.log` to ensure loss decreases below 100 within the first few epochs.
2.  **Compare Results:** Once retraining is complete, we will compare the Physics models against the `Simple ResNet` baseline to see if the physics constraints provide better generalization or lower data requirements.
3.  **Final Evaluation:** Run `launch_eval_tmux.sh` only after all models in the "In Progress" table are marked Complete.
