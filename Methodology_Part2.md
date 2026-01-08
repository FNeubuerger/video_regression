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

### 2. Hybrid Loss Function
We will train a **ResNet-UNet** architecture using a composite loss:
$$ \mathcal{L}_{total} = \mathcal{L}_{sparse} + \lambda_{PDE} \mathcal{L}_{PDE} + \lambda_{smooth} \mathcal{L}_{smooth} $$

*   **$\mathcal{L}_{sparse}$:** Mean Squared Error calculated *only* at the masked regions of M1-M4.
    $$ \mathcal{L}_{sparse} = \frac{1}{4} \sum_{i=1}^{4} || \hat{T}(p_i) - T_{GT}^{(i)} ||^2 $$
*   **$\mathcal{L}_{PDE}$:** Physics-informed loss enforcing diffusion and convection dynamics on the entire grid.
*   **$\mathcal{L}_{smooth}$:** Spatial smoothness regularization (image prior).

### 3. Verification
*   We compare the automated detection against manual inspection.
*   We test the model's interpolation capability on held-out videos.
