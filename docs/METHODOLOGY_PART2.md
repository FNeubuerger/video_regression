# Part 2: Dense Temperature Map Estimation

## Objective
The goal of this phase is to transition from single-value regression (predicting one temperature per frame) to **Dense Map Estimation**. We aim to predict a pixel-wise temperature heatmap $\hat{T}(x,y,t)$ across the entire ultrasonic frame.

## Problem Statement
We have:
1.  **Input:** Ultrasonic Video Sequence $V(x,y,t)$.
2.  **Sparse Ground Truth:** Temperature readings $T_{GT}(t)$ from 4 specific locations (M1, M2, M3, M4).
3.  **Physics Constraints:** The heat distribution should follow the Bioheat Transfer Equation (Pennes' model).

We lack **Dense Ground Truth**. Therefore, we cannot simply train a UNet with MSE loss on every pixel.

## Methodology

### 1. Data Pipeline & Sensor Localization
To use the sparse temperature readings $T_{GT}$ for supervision, we strictly align the spatial data.

**Preprocessing Pipeline (Level 0 $\to$ Level 1):**
1.  **Dynamic Active Zone Cropping:**
    *   Ultrasonic frames contain high-intensity artifacts outside the probe interface.
    *   We compute the vertical intensity projection (row-wise mean) to find the peak signal $y_{peak}$.
    *   Frames are cropped to $y_{peak} \pm 180$ px. This standardizes the input and removes noise.
2.  **Robust Sensor Detection:**
    *   Input: Temporally averaged cropped frame (mean of first 15 frames).
    *   Enhancement: CLAHE (Contrast Limited Adaptive Histogram Equalization) + Gaussian Blur.
    *   Pattern Matching: We detect all candidate blobs and select the 4-tuple that minimizes an **Axis-Aligned Rectangularity Cost**:
        $$ Cost = \Delta_{sides} + \Delta_{diagonals} + 2 \cdot (\text{Deviation from Horizontal/Vertical}) $$
    *   This ensures we identify the physical sensor grid correctly, rejecting rotated reflections.
3.  **Output:** `sensor_coordinates.json` mapping each video ID to 4 specific (x,y) coordinates.

---

## 3. Architecture Phase 2: Dense Map Estimation
We implemented `ResNetUNet`, a hybrid architecture:
*   **Encoder:** ResNet18 (pretrained) extracting hierarchical features.
*   **Decoder:** Transpose Convolutions with skip connections from the encoder to recover spatial details.
*   **Head:** 1x1 Convolution outputting a single channel heatmap.

## 4. Architecture Phase 3: Temporal Dynamics (Latent-LTC)
To better model the continuous flow of heat, we progressed to **Latent Liquid Time Constant** networks.
*   **Motivation:** The Bioheat Equation is a continuous-time PDE. Discrete RNNs struggle with the smooth decay dynamics of perfusion.
*   **Model:** `LatentLTC_UNet`.
    *   Frames are encoded to a latent vector $z$.
    *   $z$ is evolved using an LTC ODE solver.
    *   The evolved state is decoded to the heatmap.

---

## 5. Loss Function: Hybrid Physics-Informed Loss
Since we only have 4 pixels of ground truth per image, we cannot use standard MSE. We developed `BioheatHybridLoss`:

$$ L = L_{sparse-MSE} + \lambda \cdot L_{physics} $$

1.  **Sparse MSE:** Calculates error *only* at the 4 known sensor locations ($M_1 \dots M_4$).
2.  **Physics Residual:** Penalizes deviations from the Bioheat PDE across the *entire* image.
    $$ R = k\nabla^2 T - w_b c_b (T - T_a) $$
    The network is encouraged to predict a smooth, physically consistent field even where sensors are absent.

## 6. Physics Prior (Residual Learning)
To aid convergence, we use a metadata-driven prior.
*   We generate a Gaussian Heatmap based on the known Wattage and Active Zone.
*   The network learns the *residual* ($\Delta T$) rather than the absolute temperature.
*   $T_{pred} = T_{prior} + T_{network}$.

    *   `data/level1_cropped/`: Preprocessed video files.
    *   `sensor_coordinates.json`: Specific $(x,y)$ coordinates for M1-M4 in the cropped frame.

### 2. Physics Prior & Residual Learning
To guide the model given the sparse data, we incorporate strong physical priors.

**Assumption:** Heating is localized to the "Active Zone" (video center) and diffuses outwards. High heat is never expected at the far edges.

**Strategy:**
Instead of predicting $T(x,y)$ from zero, we predict a **residual correction** to a Physics Prior (Optional):
$$ \hat{T}(x,y) = T_{prior}(x,y) + \Delta T_{net}(x,y) $$
*   **$T_{prior}$**: A 2D Gaussian map centered on the Active Zone input, scaled by the known input power (default). Can be disabled ($T_{prior}=0$) to test pure data-driven learning.
*   **$\Delta T_{net}$**: The dense output from the U-Net.
*   **Benefit:** This eases the optimization landscape. The network effectively learns "how does this specific tissue/video deviate from the ideal Gaussian diffusion?"

### 3. Architecture & Hybrid Loss
**Model:** ResNet-UNet (Encoder-Decoder) with Skip Connections.
*   *Research Note:* We are exploring **Liquid Time Constant (LTC)** networks for the temporal dynamics component in future phases (see `docs/RESEARCH_PART2.md`).

**Training Objective:**
$$ \mathcal{L}_{total} = \mathcal{L}_{sparse} + \lambda_{PDE} \mathcal{L}_{PDE} + \lambda_{smooth} \mathcal{L}_{smooth} $$
*   We compare the automated detection against manual inspection.
*   We test the model's interpolation capability on held-out videos.
