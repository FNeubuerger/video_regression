# Spatial Physics Interpretability

This document explains the technical implementation and physical interpretation of the `SpatialPhysicsCNNLSTM` model, focusing on the discovery of latent tissue properties.

## 1. Overview
Traditional Physics-Informed Neural Networks (PINNs) often assume spatially uniform physical constants. However, in heterogeneous biological tissue (e.g., a tumor surrounded by healthy margins), parameters like thermal conductivity ($\beta$) and perfusion ($\alpha$) vary significantly across space.

The `SpatialPhysicsCNNLSTM` addresses this by predicting **spatial parameter maps** alongside the temperature field $T(x,y,t)$.

## 2. Inferred Parameter Maps

### Perfusion Map ($\alpha(x,y)$)
*   **Physical Meaning:** Represents the rate at which heat is removed by blood flow (the "heat sink" effect in Pennes' Bioheat equation).
*   **Discovery:** The model infers higher $\alpha$ values in regions where temperature rises more slowly than the diffusion term alone would predict. 
*   **Interpretability:** In our phantom study, higher $\alpha$ maps may identify areas with higher moisture content or localized "vessel-like" structures that mimic blood flow cooling.

### Conductivity Map ($\beta(x,y)$)
*   **Physical Meaning:** Represents thermal diffusivity ($k / \rho c$). It dictates how quickly heat spreads from high-temperature zones to low-temperature zones.
*   **Discovery:** The model observes the "blurring" or spreading of the heat front over time. Regions with faster thermal spread will show higher $\beta$ values.
*   **Interpretability:** Helps identify the boundaries between different phantom materials (e.g., tumor mimic vs. healthy mimic).

## 3. Physics Check: Verification of 4x4 Coarse Map
The model predicts temperature on a coarse 4x4 grid (16 zones). This ensures:
1.  **Gradient Stability:** Computing Laplacians on noisy 64x64 pixel data is numerically unstable. The 4x4 grid acts as a spatial low-pass filter.
2.  **Physical Consistency:** It forces the high-resolution CNN features to align with the macroscopic laws of heat transfer.
3.  **Visualization:** We can overlay these 16 zones on the original ultrasound video to see which "segments" of the tissue correspond to specific physical behaviors.

## 4. Advection and Optical Flow
By incorporating $\mathbf{v} \cdot \nabla T$, the model accounts for **tissue deformation** and **fluid motion**:
*   If the phantom expands or moves during heating, the optical flow $\mathbf{v}$ captures this velocity.
*   The model "un-distorts" the temperature change by subtracting the heat moved by physical displacement, leaving only the "pure" heating and cooling residuals.
