# XAI Integration Plan: Explainable Video Regression

This document outlines the research and implementation plan for integrating Explainable AI (XAI) into our video regression framework. The goal is to make the decision-making process of our models (CNN-LSTM, U-Net) transparent and quantitatively validatable.

## 1. Research Scope

### 1.1 Quantus (Quantitative Evaluation)
**Source:** [https://github.com/understandable-machine-intelligence-lab/Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus)

Quantus is an evaluation framework for neural network explanations. It does not generate explanations itself but evaluates how "good" an explanation is based on predefined metrics. This is critical for scientific validation.

**Key Metrics for Regression:**
*   **Faithfulness:** Does the explanation accurately reflect the features the model uses? (e.g., *Pixel Flipping*: If we mask the "hot" pixels in the explanation, does the regression error increase?)
*   **Robustness:** Are the explanations stable under slight noise?
*   **Localization:** Do the explanations align with the "Hot Spot" (Physics Prior)?

### 1.2 Target Marker / MXTDOT
**Source:** [https://github.com/DataScienceLabFHSWF/XAI_targetmarker_MXTDOT](https://github.com/DataScienceLabFHSWF/XAI_targetmarker_MXTDOT)

This reference serves as a case study for evaluating XAI in applied computer vision. The key takeaway is the pipeline structure:
1.  **Generate:** Use standard attribution methods (feature ablation, gradients).
2.  **Evaluate:** Use Quantus to score the methods.
3.  **Visualize:** overlay heatmaps on valid inputs.

### 1.3 Video Regression Nuances
Unlike classification (where we visualize "Evidence for Class Dog"), regression requires us to explain a continuous output.
*   **Attribution Target:** We cannot explain the entire $64 \times 64$ output map at once easily.
*   **Strategy:** We will define a scalar target for attribution:
    1.  **Mean Temperature:** What features drive the overall heat prediction?
    2.  **Peak Temperature:** What features drive the prediction of the hottest point (critical for safety)?
    3.  **ROI Specific:** What features drive the prediction at sensor $M_1$?

## 2. Implementation Plan

### Step 1: Dependencies
Add `captum` and `quantus` to `requirements.txt`.
*   [Captum](https://captum.ai/): For generating explanations (Integrated Gradients, LayerGradCAM, DeepLift).
*   [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus): For evaluating them.

### Step 2: Model Wrappers (`utils/xai_wrappers.py`)
Attribution libraries expect a model to output a scalar scalar. We need wrappers for our dense prediction models.

```python
class RegressionWrapper(nn.Module):
    def __init__(self, model, target_mode='mean', roi_coords=None):
        """
        target_mode: 'mean', 'max', 'roi'
        """
        ...
    def forward(self, x):
        out = self.model(x)
        if self.target_mode == 'mean': return out.mean()
        # ...
```

### Step 3: Attribution Pipeline (`evaluation/generate_explanations.py`)
Implement the generation of explanations using **LayerGradCAM** (coarse, fast) and **Integrated Gradients** (fine, slow).

### Step 4: Quantitative Evaluation (`evaluation/evaluate_xai.py`)
Implement the Quantus pipeline to answer: *"Which explanation method is most faithful for our physics-informed models?"*

### Step 5: Visualization
Generate video overlays: `[ Input RGB | Optical Flow | Prediction | Attribution Map ]`.

## 3. Targeted Questions
*   **Physics Alignment:** Do the Physics-Informed models look at "physically" relevant features (motion/convection) more than the baseline models?
*   **Input Modality:** How much does the model rely on Optical Flow vs. Raw Intensity? (We can measure this by summing attribution over the flow channels vs RGB channels).
