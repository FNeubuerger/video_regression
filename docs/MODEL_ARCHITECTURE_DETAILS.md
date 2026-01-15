# Model Architectures and Data Pipeline Documentation

## 1. Data Pipeline Overview

The temperature regression pipeline processes raw thermal video data into structured input tensors for deep learning models.

### A. Preprocessing (Level 0 -> Level 1)
- **Raw Input**: Thermal video files (e.g., `.mp4`).
- **Cropping**: Regions of Interest (ROI) are cropped to 64x64 pixels centered on the subject.
- **Artifact Masking**:  Artifacts are detected and masked out to prevent model confusion.
- **Normalization**: Pixel values are normalized (StandardScaler or MinMax) based on training set statistics.

### B. Dataset Loading (`SequenceHeatmapDataset`)
The `SequenceHeatmapDataset` class manages data loading for both scalar and spatial models.
- **Input Content**:
  - **RGB Channels**: 3 channels (Raw Thermal Data duplicated or colormap mapped).
  - **Optical Flow**: 2 channels (Flow X, Flow Y) computed between frames.
  - **Total Input Channels**: 5
- **Input Tensor Shape**: `(Batch, Time, Channels=5, Height=64, Width=64)`

### C. Targets
Depending on the model type, targets vary:
1. **Scalar Regression**: `(Batch, Time, Sensors=4)` - Temperature readings from 4 distinct sensors.
2. **Dense Regression (U-Net)**: `(Batch, Time, 1, 64, 64)` - Full spatial temperature map.

---

## 2. Model Architectures

### A. Standard Models

#### 1. SimpleResNet
- **Type**: Frame-based Scalar Regressor.
- **Backbone**: ResNet18 (modified first layer for 5 channels).
- **Temporal Handling**: Treats the sequence as a batch of independent frames or takes the last frame.
- **Output**: `(Batch, 4)` (Predicts 4 sensor values per frame).

#### 2. CNN-LSTM / PretrainedCNNLSTM
- **Type**: Spatiotemporal Scalar Regressor.
- **Backbone**: ResNet18 (feature extractor) -> LSTM (temporal aggregation).
- **Architecture**:
    - CNN extracts feature vector `(B, T, Features)`.
    - LSTM processes sequence `(B, T, Hidden)`.
    - FC Layer projects to output.
- **Output**: `(Batch, Time, 4)` or `(Batch, 4)` (Last step).

### B. Bayesian Models (Uncertainty)

#### 3. BayesianResNet / BayesianCNNLSTM
- **Type**: Probabilistic Regressor (Variational Inference).
- **Modification**: Standard Dense (Linear) layers are replaced with `BayesLinear` layers (Weights defined by $\mu, \sigma$).
- **Output**: Tuple `(Prediction, KL_Divergence)`.
    - Prediction: `(Batch, 4)`
    - KL: Scalar (Regularization term).
- **Inference**: Requires multiple forward passes (Monte Carlo Sampling) to estimate predictive uncertainty (mean/std).

### C. Dense / Physics Models

#### 4. U-Net (Dense/Spatial)
- **Type**: Image-to-Image Translation (Encoder-Decoder).
- **Encoder**: ResNet18 or Custom Conv layers (Downsampling).
- **Decoder**: Transpose Convolutions with Skip Connections (Upsampling).
- **Output**: `(Batch, 1, 64, 64)` (Temperature Map).
- **Physics**: Can integrate **Bioheat Loss** which calculates gradients ($\nabla^2 T$, $dT/dt$) directly on the output map.

#### 5. Physics-Informed Neural Networks (PINNs)
These use the standard architectures (ResNet/LSTM) but are trained with **Physics Loss functions**.
- **Bioheat Loss**: $L_{phys} = || \rho c \frac{\partial T}{\partial t} - \nabla \cdot (k \nabla T) - \omega_b \rho_b c_b (T_a - T) - q_m ||^2$
- **Components**:
    - **Convection**: Uses Optical Flow input to model heat transport by fluid motion.
    - **Metabolic**: Adds metabolic heat generation term $q_m$.
