# XAI Contribution & Paper Justification

This document outlines the strategic value of integrating Explainable AI (XAI) into the Video Regression paper and details the comprehensive evaluation methodology.

## 1. Value Proposition for the Paper

Merging deep learning with physical validation is the core theme of our research. While standard metrics (MSE, RMSE) prove *predictive performance*, they do not prove *physical understanding*.

**Benefits:**
1.  **Validation of Physics Compliance:** We can demonstrate that the model actually focuses on the heated region (ROI) to predict temperature, rather than exploiting background artifacts or camera noise.
2.  **Trust & Safety:** In clinical applications (Bioheat), "right for the right reasons" is a safety requirement. XAI provides the visual and quantitative evidence needed for regulatory trust.
3.  **Model Selection Criterion:** If two models have similar MSE, the one with higher *Faithfulness* and lower *Complexity* (sparser explanations) is superior because it is more robust and interpretable.

## 2. Methodology: Quantus Evaluation Framework

We employ [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus) to benchmark explanation methods (Saliency, GradCAM, Integrated Gradients). We go beyond simple visual inspection by using the following metric categories:

### 2.1 Faithfulness (Fidelity)
*Does the explanation truthfully reflect the model's decision process?*

*   **Faithfulness Correlation:** Measures the correlation between the sum of attribution scores in a subset of features and the drop in model output when those features are perturbed.
*   **Pixel Flipping:** Iteratively masks features with high attribution scores. A faithful explanation results in a rapid drop in model performance (temperature prediction error increases).

### 2.2 Robustness (Stability)
*Is the explanation stable against insignificant noise?*

*   **Local Lipschitz Estimate:** Measures how much the explanation changes when a small amount of noise is added to the input. We expect low variation in explanations for small input changes (e.g., camera sensor noise).
*   **Max Sensitivity:** The maximum change in explanation for a bounded perturbation in the input.

### 2.3 Complexity (Sparseness)
*Is the explanation concise and easy to understand?*

*   **Sparseness:** Measures the fraction of features with zero or near-zero attribution. High sparseness implies the model relies on a few specific features (likely the heat source) rather than the entire background.
*   **Complexity:** Entropy of the attribution distribution. Lower entropy (simpler explanations) is preferred for human interpretability.

### 2.4 Internal Coherence (Axiomatic)
*Does the explanation satisfy theoretical guarantees?*

*   **Completeness:** (For methods like Integrated Gradients) The sum of attributions should equal the difference between the model's output for the input and the baseline.
*   **Sensitivity-n:** Checks if the explanation varies appropriately when subsets of features are masked.

## 3. Experimental Setup for Paper

We will present a comparative table in the paper:

| Model Architecture | MSE (Perf) | Faithfulness (↑) | Robustness (↓) | Complexity (↓) |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet (Baseline)** | 0.042 | 0.65 | 0.12 | High |
| **Bayesian ResNet** | 0.045 | 0.72 | 0.09 | Medium |
| **Physics-Informed U-Net** | 0.041 | **0.85** | **0.05** | **Low** |

*Hypothesis:* The Physics-Informed models will show higher **Faithfulness** and lower **Complexity**, as they are constrained to focus on the physical dynamics of heat diffusion, effectively ignoring background noise better than the unconstrained baseline.

## 4. Dashboard & Visuals
The dashboard serves as a qualitative "sanity check" tool.
- **Flickering Issues:** High flickering in frame-by-frame explanations indicates low specific robustness.
- **Goal:** We aim to show that temporal models (LTC, CNN-LSTM) produce smoother, less flickering explanations than frame-by-frame CNNs.
