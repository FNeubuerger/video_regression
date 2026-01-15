# Benchmark Status Report

**Date:** January 15, 2026
**Overall Status:** **ACTIVE.** Three parallel training streams are effectively running. Critical code fixes for `train_all_models.py` (target dimension mismatch) and `train_unet_sparse.py` (unpacking error) have been deployed.

## 1. Active Training Streams

We currently have four parallel training streams active (Verified 2026-01-15):

| Stream | Component | Models Included | Status | Log Location |
| :--- | :--- | :--- | :--- | :--- |
| **1. Standard** | `train_all_models.py` | `CNNLSTM` `SpatialPhysicsCNNLSTM` | 🚀 **Running** (Epoch 9/50) | `logs/retrain/restart_main.log` |
| **2. Uncertainty** | `train_uncertainty.py` | `Ensemble` (5x ResNet) | 🚀 **Running** (Epoch 5/20) | `logs/retrain/restart_uncertainty.log` |
| **3. Dynamics** | `restart_ltc_benchmarks.sh` | `ConvLTC` `LatentLTC` `LatentLTC-Var` | 🚀 **Running** (Epoch 1/30) | `logs/ltc/` |
| **4. Physics** | `run_physics_benchmarks.sh` | **Bioheat Variants** (Scalar & Spatial) + Bayesian | 🚀 **Running** (Epoch 1-2/30) | `logs/physics/` |

---

## 2. Completed Benchmarks (Phase 1)

These models have successfully finished training on the new `level1_cropped` dataset.

| Category | Model Name | Validation Loss (MSE) | Epochs | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Standard** | `CNNLSTM` | **1.1842** | 50 | High loss compared to Pretrained. |
| **Standard** | `PretrainedCNNLSTM` | **0.4609** | 32 | Best performing standard model so far. |
| **Standard** | `SimpleResNet` | **0.4723** | 47 | Strong baseline for scalar regression. |
| **Physics** | `PhysicsCNNLSTM` | **2.4260** | 43 | High loss likely due to strong regularization. |

---

## 3. Pending / Queued Benchmarks

The following models are identified in the Plan but are not yet actively training or need integration.

### A. Advanced Bioheat Physics (Scalar)
*Required Scripts:* `train_*_bioheat.py`
- [ ] `Bioheat PINN` (`train_bioheat.py`)
- [ ] `Convection Bioheat` (`train_convection_bioheat.py`)
- [ ] `Metabolic Bioheat` (`train_metabolic_bioheat.py`)
- [ ] `Spatial Bioheat` (`train_spatial_bioheat.py`)
- [ ] `Spatial Convection Bioheat` (`train_spatial_convection_bioheat.py`)
- [ ] `Spatial Metabolic Bioheat` (`train_spatial_metabolic_bioheat.py`)

### B. Bayesian Physics (Scalar)
*Required Scripts:* `train_bayesian_*.py`
- [ ] `Bayesian PINN`
- [ ] `Bayesian CNNLSTM`
- [ ] `Bayesian Convection PINN`
- [ ] `Bayesian Metabolic PINN`
- [ ] `Bayesian Spatial Convection`

### C. Dense Estimation (U-Nets)
*Required Scripts:* `train_unet_sparse.py`, `train_unet_hybrid.py`
- [ ] **Standard U-Net** (`train_unet_sparse.py`) - *Code Fixed*
- [ ] **Variational U-Net** (`train_unet_sparse.py --variational`) - *Implemented & Verified*
- [ ] **Hybrid U-Net** (`train_unet_hybrid.py`)

---

## 4. Evaluation & Next Steps

### Action Items
1.  **Monitor Streams:** Check logs for early convergence or NaN errors in strict physics models.
2.  **Launch U-Net Stream:** Once compute frees up (or if parallel capacity allows), launch the U-Net benchmarks using the newly fixed script.
3.  **Scientific Plotting:** Prepare comparison plots for `PretrainedCNNLSTM` vs `SimpleResNet`.

### Known Issues Resolved
- **Dimension Mismatch:** Fixed `CNNLSTM` requesting `[B, 4]` but getting `[B, T, 4]`.
- **Unpacking Error:** Fixed data loader returning tuples > 2 elements (e.g. masks, artifact masks) which crashed scalar training loops.
- **Variational Logic:** Added full KLD support to `train_unet_sparse.py`.

