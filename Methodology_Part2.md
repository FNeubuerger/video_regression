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

### 1. Sensor Localization (Current Step)
To use the sparse temperature readings $T_{GT}$ for supervision, we must know **where** in the image these readings correspond to.
*   **Algorithm:** Automated blob detection to find the 4 sensors in the video.
*   **Constraint:** The sensors form a rectangular grid (Top-Left, Top-Right, Bottom-Left, Bottom-Right).
*   **Output:** `sensor_coordinates.json` containing $(x,y)$ and radius $r$ for M1-M4.

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
