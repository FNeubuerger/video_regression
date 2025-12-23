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
