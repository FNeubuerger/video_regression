# TODO: Fix Spatial Model Crashes

## Issue
The spatial models (`spatial_bioheat_resnet`, `spatial_convection_bioheat`, `spatial_metabolic_bioheat`) were crashing during the validation phase of training.

**Error:**
`RuntimeError: The size of tensor a (4) must match the size of tensor b (32) at non-singleton dimension 1`

**Cause:**
The `AdvancedBioHeatLoss` function expects inputs with a temporal dimension (e.g., `(B, T, H, W)`). In the validation loop, the model predictions `(B, 4, 4)` and labels `(B, 1)` were passed directly without adding a dummy temporal dimension. The loss function misinterpreted the spatial dimension `H=4` as the temporal dimension `T=4`, causing a shape mismatch with the labels `(B, 1)`.

## Fix
Updated the validation loops in the following files to unsqueeze predictions and labels, adding a dummy temporal dimension (size 1):
- `training/train_spatial_bioheat.py`
- `training/train_spatial_convection_bioheat.py`
- `training/train_spatial_metabolic_bioheat.py`

For `spatial_convection_bioheat` and `spatial_metabolic_bioheat`, we also ensured that the optical flow is extracted, downsampled, and passed to the loss function during validation, consistent with the training loop.

## Next Steps
- [x] Fix validation loop in `training/train_spatial_bioheat.py`
- [x] Fix validation loop in `training/train_spatial_convection_bioheat.py`
- [x] Fix validation loop in `training/train_spatial_metabolic_bioheat.py`
- [ ] Rerun the benchmarks to verify the fix.

## Priority Actions (Jan 9)
- [ ] **Monitor Benchmarks:** Check completion of `ConvLTC` and `U-Net Hybrid` (currently Epoch ~17 and ~10).
- [ ] **Core Evaluation:** Run `evaluation/evaluate_dense.py` on the completed Part 3 models.
    - Compare `ConvLTC` vs `U-Net Hybrid` vs `Latent LTC`.
    - Generate tables for the paper.
- [ ] **Paper Writing:** Draft the Results section using the new benchmark data.

## Part 2/3 (Dense & LTC) Analysis
- [ ] **Implementation:** Create `evaluation/evaluate_dense.py` for dense map metrics (SSIM, Hausdorff, IoU).
- [ ] **Bayesian U-Net:** Implement "Probabilistic Bottleneck" (Variational Layer) in `models/dense_heads.py`.
- [ ] **Bayesian LTC:** Implement Variational Encoder for `LatentLTC_UNet`.
- [ ] **Evaluation:** Run comparative analysis between Physics-Informed U-Net and Latent LTC.

## XAI & Interpretability (Bonus / Deprioritized)
- [x] **Justification:** Created `research/XAI_PAPER_CONTRIBUTION.md` (on `feature/xai-integration`).
- [x] **Metric Implementation:** `evaluation/evaluate_xai.py` implemented (on `feature/xai-integration`).
- [ ] **Fix Memory Error:** `FaithfulnessCorrelation` causing OOM. (Low Priority)
- [ ] **Dashboard Fix:** `demo/xai_dashboard.py` flickers. (Low Priority)


## XAI & Interpretability (Bonus / Deprioritized)
- [x] **Justification:** Created research/XAI_PAPER_CONTRIBUTION.md (on xai branch).
- [x] **Metric Implementation:** evaluation/evaluate_xai.py implemented (on xai branch).
- [ ] **Fix Memory Error:** FaithfulnessCorrelation causing OOM. (Low Priority)
- [ ] **Dashboard Fix:** demo/xai_dashboard.py flickers. (Low Priority)
