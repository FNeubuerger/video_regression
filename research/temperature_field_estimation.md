# Research Summary: Dense Temperature Map Estimation from Sparse Sensors

## 1. Problem Definition
*   **Input:** Grayscale/RGB video frames (possibly with Optical Flow).
*   **Sparse Data:** 4 discrete temperature sensors arranged in a square pattern.
*   **Goal:** Estimate a dense, pixel-wise temperature heatmap ($H \times W$) for the "active zone".
*   **Constraint:** "Ground Truth" dense maps do not exist; only sparse points are known.

## 2. Recommended Architecture: ResNet-UNet (Encoder-Decoder)
To transition from scalar regression to dense map regression, the architecture must move from a "Funnel" (Encoder -> Vector) to an "Hourglass" (Encoder -> Bottleneck -> Decoder).

### Architecture Choice: U-Net / ResNet-UNet
*   **Why:** U-Net is the standard for image-to-image translation tasks (segmentation, depth estimation, field estimation) because **Skip Connections** preserve spatial information lost during pooling.
*   **Implementation Strategy:**
    *   **Encoder:** Reuse the **Simple ResNet** backbone (already trained and performant). Freeze the early layers (Layer 1-3) to retain learned feature extraction.
    *   **Decoder:** Add a sequence of `ConvTranspose2d` (Upsampling) layers + `ReLU` + `BatchNorm` to project the 512-dim feature vector back to $64 \times 64$ spatial resolution.
    *   **Skip Connections:** Link the output of `model.layer1`, `model.layer2`, etc., to the corresponding decoder layers to sharpen the "hot spot" boundaries.

### Alternative: Fully Convolutional Network (FCN)
*   Simpler but coarser outputs. Better if only a "blob" location is needed, but U-Net is preferred for accurate gradient mapping.

## 3. Handling Sparse Ground Truth (The "Interpolation" Problem)
Since we cannot train with a dense pixel-wise MSE loss (we don't have the targets), we must guide the network using **Hybrid Constraints**.

### A. Geometric Interpolation (Baseline)
Use **Radial Basis Functions (RBF)** or **Bilinear Interpolation** to generate a synthetic "Smooth Ground Truth" from the 4 points.
*   *Pros:* Easy to implement.
*   *Cons:* Ignores physics (heat doesn't always flow in perfect circles).
*   *Proposed Use:* Use this as a **Pre-training Target**. Train the U-Net to replicate RBF interpolation first to learn the mechanics of generating a map.

### B. Physics-Informed Training (PINN Method)
This is the robust solution. We define a loss function that relies on the Physics, not dense labels.
$$ Loss_{Total} = Loss_{Data} + \lambda \cdot Loss_{Physics} $$

1.  **Data Loss ($Loss_{Data}$):**
    *   Masked MSE. We take the predicted heatmap $\hat{Y}$, extract the values at the **4 pixel coordinates** corresponding to the sensors, and minimize the error against the real sensor readings.
    *   $Loss = \sum_{i=1}^{4} ||\hat{Y}(x_i, y_i) - T_{sensor_i}||^2$

2.  **Physics Loss ($Loss_{Physics}$):**
    *   Apply the **Bioheat / Heat Diffusion Equation** to the **entire predicted map**.
    *   The network learns that "between the sensors, the temperature must change smoothly according to diffusion."
    *   We already have `AdvancedBioHeatLoss` – this is perfectly suited for this task.

## 4. Transfer Learning Plan
1.  **Branch Repository:** Create `feature/dense-map-estimation`.
2.  **Data Loader Update:**
    *   Parse the new CSV format (4 columns + timestamps).
    *   Identify pixel coordinates of the 4 sensors (using the "bright spots" in the video).
    *   Return: `Image`, `[(x1,y1,t1), (x2,y2,t2), ...]`.
3.  **Model Adaptation:**
    *   Load `SimpleResNet` weights.
    *   Discard `fc` layer.
    *   Attach `Decoder` module.
4.  **Training Phases:**
    *   **Phase 1 (Geometric):** Train Decoder to match RBF interpolated maps (Stabilization).
    *   **Phase 2 (Physics):** Fine-tune using Masked MSE (at 4 points) + Bioheat Loss (everywhere).

## 5. References
*   *Ronneberger, O., et al.* "U-Net: Convolutional Networks for Biomedical Image Segmentation." (MICCAI 2015).
*   *Raissi, M., et al.* "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." (Journal of Computational Physics, 2019).
