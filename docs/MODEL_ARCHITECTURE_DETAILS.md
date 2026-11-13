# Model Architectures and Data Pipeline Documentation

This document reflects the current implementation in the repository as of now. It is intentionally aligned with the actual code, not with earlier design notes.

## 1. Data pipeline

The training pipeline loads short video clips, converts them to tensors, and passes them to scalar or spatial regression models.

### A. Input data and indexing

`SequenceHeatmapDataset` in `utils/sequence_dataset.py` reads videos from `data/level1_cropped` and matching sensor logs from `data/level0_raw`.

- Each clip yields `sequence_length` frames (default: `16`, configurable via dataset args).
- Frames are resized to `target_size=(64, 64)`.
- Optional optical flow can be appended, yielding `C=5` instead of `C=3`.
- A sparse temperature target is generated for each frame at the sensor positions.
- The target scalar vector is four sensor values (`M1`..`M4`).

### B. Tensor conventions in the current code

The dataset returns tensors in this form:

- `frames_t`: `(sequence_length, C, H, W)`
- `targets_t`: `(sequence_length, 1, H, W)`
- `priors_t`: `(sequence_length, 1, H, W)`
- `scalars_t`: `(sequence_length, 4)`

When a model expects a batch dimension, the common shapes are:

- Sequence model input: `(B, T, C, H, W)`
- Single-frame model input: `(B, C, H, W)`

### C. Output targets

The code supports multiple target types:

1. Scalar regression for 4 sensors
   - Output shape examples: `(B, 4)` or `(B, T, 4)`
2. Dense / spatial regression maps
   - Output shape examples: `(B, 1, H, W)`, `(B, T, H, W)` or `(B, T, 1, H, W)` depending on model
3. Bayesian outputs
   - Usually `(prediction, kl_loss)`

---

## 2. Current model inventory

The repository currently contains several model families. The registry in `utils/model_registry.py` exposes them as named entries.

### A. Standard scalar models

#### 1. `SimpleResNet`
- Type: frame-based scalar regressor
- Backbone: pretrained `ResNet18`
- Adaptation: first conv layer is replaced to accept non-RGB channels
- Temporal behavior: if input is 5D `(B, T, C, H, W)`, it takes the last frame only
- Output: `(B, 4)`
- Location: `models/backbones.py`

#### 2. `CNNLSTM`
- Type: spatiotemporal scalar regressor
- Architecture: CNN feature extraction per frame -> LSTM over time -> final regression head
- Input: `(B, T, C, H, W)`
- Output: `(B, 4)`
- Important detail: the forward pass uses the last LSTM state and then applies the FC head

#### 3. `PretrainedCNNLSTM`
- Same concept as `CNNLSTM`, but the spatial encoder is built from a pretrained CNN backbone
- Accepts non-RGB input by replacing the first conv layer
- Output: `(B, 4)`

### B. Physics and sequence-aware models

#### 4. `PhysicsCNNLSTM`
- Type: sequence model for scalar temperature regression
- Architecture: CNN -> LSTM -> per-frame regression head
- Output shape: `(B, T, 4)`
- This is the implementation that is closest to a physics-informed temporal model, even though the actual physics terms are not enforced in this class itself

#### 5. `SpatialPhysicsCNNLSTM`
- Type: advanced spatial-temporal model with map outputs
- Architecture: CNN feature extraction + temporal LSTM + per-frame map decoder
- Output:
  - `temp_map`: `(B, T, 4, 4)`
  - `alpha_map`: `(B, 1, 4, 4)`
  - `beta_map`: `(B, 1, 4, 4)`
- This is a more experimental model than the regular scalar sequence regressors

### C. Bayesian models

#### 6. `BayesianResNet`
- Type: Bayesian scalar regressor
- Backbone: `ResNet18`
- Final linear layers are Bayesian (`torchbnn.BayesLinear`)
- If input is 5D, it takes the last frame before the backbone
- Output: `(predictions, kl_loss)`
- `predictions`: `(B, 4)`
- `kl_loss`: scalar KL term for variational regularization

#### 7. `BayesianCNNLSTM`
- Type: Bayesian temporal regressor
- Backbone + LSTM + Bayesian output head
- Input: `(B, T, C, H, W)`
- Output: `(predictions, kl_loss)`
- `predictions`: `(B, T, 4)`

#### 8. `FullBayesianResNet` / `BayesianSpatialResNet`
- Present in the model registry and bayesian module, but not the primary training path described in the basic docs
- They are additional probabilistic extensions of the resnet family

### D. Dense / spatial models

#### 9. `ResNetUNet`
- Type: U-Net variant with ResNet18 encoder
- Architecture: ResNet18 encoder + decoder with skip connections
- Input: usually `(B, C, H, W)`; 5D inputs are reduced to the last frame
- Output: `(B, 1, H, W)`
- If `variational=True`, it returns `(output, kl_loss)`
- Location: `models/dense_heads.py`

#### 10. `SpatialResNet`
- Type: spatial map predictor
- Backbone: ResNet18 with FC and AvgPool removed to keep spatial information
- Decoder: 1x1 convs and upsampling
- Input: `(B, C, H, W)` or `(B, T, C, H, W)`
- Output:
  - if input is spatial: `(B, H, W)`
  - if input is sequential: `(B, T, H, W)`

### E. Experimental / advanced models

#### 11. `ConvLTC_Model`
- Located in `models/conv_ltc.py`
- A more advanced continuous-time / latent dynamics model
- Not part of the core scalar CNN/LSTM family

#### 12. `LatentLTC_UNet`
- Located in `models/latent_ltc.py`
- Sequence-aware latent model with U-Net-like structure
- Registered in `MODEL_REGISTRY` as a later/experimental model

---

## 3. Important implementation notes

1. The project does not enforce a single canonical architecture. Multiple families coexist in parallel:
   - simple scalar regressors
   - temporal CNN/LSTM models
   - Bayesian variants
   - dense U-Net outputs
   - experimental continuous-time models

2. In the current implementation, many models accept 5D sequence tensors but internally reduce them to a single frame for image-based models. This is especially true for `SimpleResNet`, `BayesianResNet`, and `ResNetUNet`.

3. The dataset and model registry support optional optical flow input. When enabled, the channel count becomes 5, but the actual handling differs by model: some models accept it directly as an input channel, others are adapted via the first conv layer.

4. The repo is best understood as a modular experimental testbed for temperature regression, not as a single final architecture with a strict contract.

---

## 4. Short summary

The current codebase is consistent with a multi-branch model zoo:

- `SimpleResNet` / `BayesianResNet`: single-frame scalar regression
- `CNNLSTM` / `PretrainedCNNLSTM` / `PhysicsCNNLSTM`: temporal scalar regression
- `BayesianCNNLSTM`: Bayesian temporal scalar regression
- `ResNetUNet`: dense spatial output
- `SpatialResNet` / `SpatialPhysicsCNNLSTM`: spatial map prediction
- `ConvLTC_Model` / `LatentLTC_UNet`: experimental advanced sequence models

This is broader and more heterogeneous than the original single-architecture description, but it matches the actual implementation in the repository.
