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
3.  **Output:**
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
