# Benchmark Status Report

**Date:** January 15, 2026
**Overall Status:** **ACTIVE (All Streams Recovered).** All parallel training streams are effectively running, including the previously failed Bayesian Physics stream.

## 1. Active Training Streams

We currently have **six** parallel training streams active (Verified 2026-01-15 14:15):

| Stream | Component | Models Included | Status | Log Location | WandB Project |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Standard** | `train_all_models.py` | `CNNLSTM` | 🚀 **Running** ([Link](https://wandb.ai/dsfhswf/video-temperature-regression/runs/b01ipl91)) | `logs/cnnlstm.log` | `video-temperature-regression` |
| **2. Uncertainty** | `train_uncertainty.py` | `Ensemble` (5x ResNet) | 🚀 **Running** ([Link](https://wandb.ai/dsfhswf/video-temperature-regression/runs/49y89rvb)) | `logs/ensemble.log` | `video-temperature-regression` |
| **3. Dynamics** | `restart_ltc_benchmarks.sh` | `ConvLTC` `LatentLTC` | 🚀 **Running** (Restarted) | `logs/ltc/` | `video-regression-part3` |
| **4. Physics** | `run_physics_benchmarks.sh` | **Bioheat Variants** (Scalar & Spatial) | 🚀 **Running** | `logs/physics/` | `video-temperature-regression` |
| **5. Bayesian Physics** | `launch_stream5_bayesian.sh` | `Bayesian PINN` variants (x4) | 🚀 **Running** (Recovered) | `logs/part3/` | `video-temperature-regression` |
| **6. Dense (U-Net)** | `run_missing_benchmarks.sh` | `Standard`, `Variational`, `Hybrid` | 🚀 **Running** | `logs/unet/` | `video-regression-part2` |

> **Note on WandB**: Models are split across three projects due to legacy configuration.
> *   **Main Project**: `video-temperature-regression` (Standard, Physics, Bayesian, Uncertainty)
> *   **Dense Project**: `video-regression-part2` (U-Net)
> *   **LTC Project**: `video-regression-part3` (ConvLTC, LatentLTC)
>
> **LTC Run Names**: `conv_ltc_benchmark`, `latent_ltc_benchmark`, `latent_ltc_variational` (Look for these in `video-regression-part3`)
>
> **Direct WandB Links (LTC):**
> *   [ConvLTC Benchmark](https://wandb.ai/dsfhswf/video-regression-part3/runs/55lausyk)
> *   [LatentLTC Benchmark](https://wandb.ai/dsfhswf/video-regression-part3/runs/9v4yiscw)
> *   [LatentLTC Variational](https://wandb.ai/dsfhswf/video-regression-part3/runs/vfj74m2u)

---

## 2. Completed Benchmarks (Phase 1)

These models have successfully finished training on the new `level1_cropped` dataset.

| Category | Model Name | Validation Loss (MSE) | Epochs | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Standard** | `CNNLSTM` | **1.1842** | 50 | High loss compared to Pretrained. (Single Scalar Output) |
| **Standard** | `PretrainedCNNLSTM` | **0.4609** | 32 | Best performing standard model. (Single Scalar Output) |
| **Standard** | `SimpleResNet` | **0.4723** | 47 | Strong baseline. (Single Scalar Output) |
| **Physics** | `PhysicsCNNLSTM` | **2.4260** | 43 | High loss likely due to regularization. (Single Scalar Output) |

> **Note:** All completed benchmarks listed above were trained to predict a **single scalar temperature** value (averaging the 4 sensor readings or training on a single aggregated target). The newer Bayesian and U-Net streams are being configured for **4-sensor vector output** and **spatial map output** respectively. This distinction is crucial for comparing MSE across streams.

---

## 3. Pending / Queued Benchmarks

The following models are identified in the Plan but are not yet actively training or need integration.

### A. Advanced Bioheat Physics (Scalar)
*Required Scripts:* `train_*_bioheat.py`
- [x] `Bioheat PINN` (`train_bioheat.py`) - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-temperature-regression/runs/5ctpq03n)) (Restarted)
- [x] `Convection Bioheat` (`train_convection_bioheat.py`) - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-temperature-regression/runs/kpakkxh7)) (Restarted)
- [x] `Metabolic Bioheat` (`train_metabolic_bioheat.py`) - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-temperature-regression/runs/w995g5f6)) (Restarted)
- [x] `Spatial Bioheat` (`train_spatial_bioheat.py`) - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-temperature-regression/runs/1lcj60y4)) (Restarted)
- [x] `Spatial Convection Bioheat` (`train_spatial_convection_bioheat.py`) - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-temperature-regression/runs/iedwiw1r))
- [x] `Spatial Metabolic Bioheat` (`train_spatial_metabolic_bioheat.py`) - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-temperature-regression/runs/hx2cvsj3)) (Restarted)

### B. Bayesian Physics (Scalar)
*Required Scripts:* `train_bayesian_*.py`
- [x] `Bayesian PINN` - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-temperature-regression/runs/8tbuxmnj)) confirmed running
- [x] `Bayesian CNNLSTM` - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-temperature-regression/runs/3bqfroj5)) (Restarted)
- [x] `Bayesian Convection PINN` - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-temperature-regression/runs/ilcdl6lz)) confirmed running
- [x] `Bayesian Metabolic PINN` - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-temperature-regression/runs/j22n8mox)) confirmed running
- [x] `Bayesian Spatial Convection` - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-temperature-regression/runs/k21c5la3)) confirmed running

### C. Dense Estimation (U-Nets)
*Required Scripts:* `train_unet_sparse.py`, `train_unet_hybrid.py`
- [x] **Standard U-Net** (`train_unet_sparse.py`) - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-regression-part2/runs/z8gojzig)) confirmed running
- [x] **Variational U-Net** (`train_unet_sparse.py --variational`) - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-regression-part2/runs/k8l1opvk)) confirmed running 
- [x] **Hybrid U-Net** (`train_unet_hybrid.py`) - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-regression-part2/runs/k0916o5l))confirmed running

### D. Dynamics (LTC)
*Required Scripts:* `train_ltc.py`
- [x] **ConvLTC** - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-regression-part3/runs/82p64kh8))
- [x] **LatentLTC** - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-regression-part3/runs/iv2ds7xa))
- [x] **LatentLTC Variational** - 🚀 **Running** ([WandB Link](https://wandb.ai/dsfhswf/video-regression-part3/runs/9wi3kp7m))

---

## 4. Evaluation & Next Steps

### Action Items
1.  **Monitor Streams:** Check logs for early convergence or NaN errors in strict physics models.
2.  **Fix Bayesian Physics Stream:** The Bayesian models are failing due to shape mismatches `[32, 5]` vs `input 640` (likely flattened sequence) and `tuple` attribute errors (BayesianResNet outputting tuple `(pred, kl)`). These need immediate attention.
3.  **Scientific Plotting:** Prepare comparison plots for `PretrainedCNNLSTM` vs `SimpleResNet`.

### Known Issues
- **Bayesian PINN Mismatch:** Input shape handling for sequence data seems incorrect in the Bayesian script refactor.
- **Tuple Output:** Evaluation or training loops expecting single tensor output but receiving `(pred, kl)` tuple from `BayesianResNet`.

### Resolved
- **Bayesian Physics Stream:** Fixed return values (tuple unpacking errors) and shape mismatches in spatial bayesian models.
- **LTC Stream:** Restarted benchmarks after identifying silent failure.
- **Dimension Mismatch (Standard):** Fixed `CNNLSTM` requesting `[B, 4]` but getting `[B, T, 4]`.
- **Unpacking Error:** Fixed data loader returning tuples > 2 elements.
- **Variational Logic:** Added full KLD support to `train_unet_sparse.py`.

