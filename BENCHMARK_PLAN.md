# Video Temperature Regression: Benchmark & Comparison Plan (Dataset Update 2026)

**CRITICAL UPDATE (Jan 14, 2026):** All benchmarks are being reset. We identified that previous models were trained on legacy/raw data. A full migration to `data/level1_cropped` has been completed. The plan below reflects the new rigorous retraining protocol.

This document outlines the suite of 14 active benchmarks and the specific comparative analyses designed to evaluate the contributions of physical constraints, temporal modeling, and uncertainty quantification.

## 1. Active Benchmarks Overview

We are currently retraining **20+ distinct models** on the new 5D Sequence dataset (`level1_cropped`). Comparison against legacy results will be performed to quantify dataset quality improvements.

### A. Temporal Models (Sequence-based)
*Input: Sequence of 5 Frames (RGB + Optical Flow)*
- [ ] **CNNLSTM** (Pending Retrain)
- [ ] **Pretrained CNNLSTM** (Pending Retrain)
- [ ] **Physics CNNLSTM** (Pending Retrain)
- [ ] **Bioheat PINN** (Pending Retrain)
- [ ] **Convection Bioheat** (Pending Retrain)
- [ ] **Metabolic Bioheat** (Pending Retrain)

### B. Spatial Models (Frame-based)
*Input: Single Frame (RGB + Optical Flow)*
- [ ] **Simple ResNet** (Pending Retrain)
- [ ] **Spatial Bioheat** (Pending Retrain)
- [ ] **Spatial Convection** (Pending Retrain outputting 4x4 maps)
- [ ] **Spatial Metabolic** (Pending Retrain outputting 4x4 maps)

### C. Dense Map Models (Part 2: U-Net)
*Input: Frame/Sequence with Dense Output ($H \times W$)*
- [ ] **U-Net Baseline** (Pending Integration)
- [ ] **U-Net + Physics Prior** (Pending Integration)
- [ ] **Hybrid U-Net** (Pending Integration)

### D. Time & Dynamics (Part 3: LTC/ODES)
- [ ] **ConvLTC** (Pending)
- [ ] **Latent LTC U-Net** (Pending)

### E. Uncertainty Models
- [ ] **Ensemble**
- [ ] **Bayesian Head**
- [ ] **Full Bayesian** (ResNet)
- [ ] **Bayesian PINN**
- [ ] **Bayesian CNNLSTM**
- [ ] **Bayesian U-Net**
- [ ] **Bayesian LTC**


---

## 2. Comparative Analysis Plan

We will perform the following pairwise and group comparisons to validate our hypotheses.

### Comparison 1: Scalar vs. Dense Estimation (Part 1 vs. Part 2)
*Hypothesis: Estimating the full temperature field regularizes the scalar prediction (max temp) better than direct regression.*

| Model A | Model B | Comparison Goal |
| :--- | :--- | :--- |
| **Simple ResNet** | **U-Net (Max Pool)** | Does predicting the map then taking the max beat direct regression? |
| **U-Net** | **Hybrid U-Net** | Does PDE regularization improve hotspot localization (Hausdorff)? |

### Comparison 2: The "Ladder of Physics" (Temporal)
*Hypothesis: Adding more detailed physical processes improves accuracy and generalization.*

| Model A | Model B | Comparison Goal |
| :--- | :--- | :--- |
| **Pretrained CNNLSTM** | **Bioheat PINN** | Does the basic Bioheat equation (Diff+Perf) outperform pure data-driven learning? |
| **Bioheat PINN** | **Convection Bioheat** | Does explicitly modeling fluid dynamics (blood flow) via Optical Flow help? |
| **Convection Bioheat** | **Metabolic Bioheat** | Does learning a metabolic heat source term ($Q_{met}$) add value? |

### Comparison 3: Time Dynamics (LTC vs. LSTM)
*Hypothesis: Continuous-time models (LTCs) handle irregular sampling and physics dynamics better than discrete RNNs.*

| Model A | Model B | Comparison Goal |
| :--- | :--- | :--- |
| **CNNLSTM** | **ConvLTC** | Discrete vs. Continuous recurrence for diffusion processes. |
| **Latent LTC** | **Hybrid U-Net** | Does explicitly modeling latent dynamics over time beat frame-by-frame estimation? |

### Comparison 4: Shortcut Learning vs. Real Learning (Masking)
*Hypothesis: Models trained on raw video "cheat" by using bright sensor/antenna artifacts as shortcuts. Masking reveals the true performance on soft tissue features.*

| Model A | Model B | Comparison Goal |
| :--- | :--- | :--- |
| **Simple ResNet** | **ResNet (Masked)** | Quantify accuracy drop when shortcuts are removed. |
| **Bioheat PINN** | **PINN (Masked)** | Does the physics prior compensate for the loss of sensor visual cues? |
| **Unmasked GradCAM** | **Masked GradCAM** | Visual proof of attention shift from sensors to tissue. |

---

## 4. Inverse Physics & Generalization Validation

### Section 4.1: Spatial Parameter Discovery (Inverse Modeling)
*Hypothesis: Learning heterogeneous tissue properties ($\alpha, \beta$) spatially reveals subsurface structures and improves PDE adherence.*

| Model | Novelty | Comparison Goal |
| :--- | :--- | :--- |
| **Convection Bioheat** | Scalar $\alpha, \beta$ | Baseline physics with uniform parameters. |
| **Spatial Physics CNNLSTM** | $4\times 4$ Property Maps | Does discovering tissue heterogeneity improve local temp accuracy? |

### Section 4.2: LOSO Cross-Validation
*Hypothesis: Random-split validation overestimates performance; Leave-One-Sequence-Out (LOSO) reveals true generalization across subjects.*

1.  **Baseline vs. Spatial Physics**: Compare standard CNN-LSTM against Inverse Physics model under LOSO.
2.  **Generalization Gap**: Quantify $| RMSE_{random} - RMSE_{LOSO} |$ to establish the "Reliability Score" of physics-informed models.


## 5. Evaluation Metrics & Strategy

### A. Scalar Metrics (Regression)
1.  **RMSE/MAE**: Accuracy of Peak Temperature ($T_{max}$).
2.  **Calibration Error (ECE)**: Reliability of uncertainty bounds.

### B. Dense Map Metrics (Field Estimation)
1.  **Pixel-wise RMSE**: Accuracy over the entire field.
2.  **SSIM (Structural Similarity)**: Perceptual quality of the heat diffusion pattern.
3.  **Hausdorff Distance (Thresholded)**: Safety metric. Distance between predicted and true "Safe Zone" ($T < 43^\circ C$) boundaries.
4.  **IoU (Intersection over Union)**: Overlap of the "Tumor Ablation Zone" ($T > 50^\circ C$).

---

## 4. Next Steps & Research Actions

### Implementation Tasks
1.  **Implement Dense Evaluation Script:**
    *   Create `evaluation/evaluate_dense.py`.
    *   Implement **SSIM**, **Hausdorff**, and **IoU** metrics.
2.  **Bayesian Part 2 (U-Net):**
    *   **Strategy:** Do NOT implement Full Bayesian U-Net (too heavy).
    *   **Implementation:** Implement **Probabilistic Bottleneck** (Encoder $\to$ Variational Layer $\to$ Decoder) or use MC-Dropout.
3.  **Bayesian Part 3 (LTC):**
    *   **Strategy:** Implement **Variational Encoder** with Deterministic LTC (Latent ODE approach).
    *   **Avoid:** Bayesian weights inside the stiff ODE solver (instability risk).

---

## 5. Final Evaluation Execution

Once training is complete (approx. 50 epochs), you can run evaluations using the provided tools.

### Comparison Tables
```bash
python evaluation/generate_tables.py --include_dense
```

### Dense Map Visualization
```bash
python evaluation/evaluate_dense.py --model unet_hybrid --visualize
```

## 4. Final Evaluation Execution

Once training is complete (approx. 50 epochs), you can run all evaluations in parallel using the provided tmux script:

```bash
./launch_eval_tmux.sh
```

This will create a tmux session named `video_regression_eval` with 5 windows running the evaluations in parallel groups.

Alternatively, you can run them sequentially or with a specific job limit using:

```bash
./run_final_evaluations.sh 2  # Runs 2 jobs at a time
```

## 5. Result Aggregation

After running the evaluations, generate the final comparison tables using:

```bash
python evaluation/generate_tables.py
```

This will produce:
1.  `results/tables/comprehensive_results.csv`: All models sorted by category and RMSE.
2.  `results/tables/{category}_results.csv`: Individual tables for Temporal, Spatial, and Uncertainty models.
3.  `results/tables/{category}_results.tex`: LaTeX code for the paper.
### Detailed Command List (Reference)

### A. Temporal Models (Deterministic)
*Note: NLL will be NaN for these models.*

```bash
# 1. CNNLSTM
python evaluation/comprehensive_uncertainty_eval.py --model CNNLSTM --checkpoint models/cnnlstm_model.pth --samples 1

# 2. Pretrained CNNLSTM
python evaluation/comprehensive_uncertainty_eval.py --model PretrainedCNNLSTM --checkpoint models/pretrained_cnnlstm_model.pth --samples 1

# 3. Physics CNNLSTM
python evaluation/comprehensive_uncertainty_eval.py --model PhysicsCNNLSTM --checkpoint models/physics_cnnlstm_model.pth --samples 1

# 4. Bioheat PINN (Advanced Bioheat)
python evaluation/comprehensive_uncertainty_eval.py --model SpatialPhysicsCNNLSTM --checkpoint models/advanced_bioheat_model.pth --samples 1

# 5. Convection Bioheat
python evaluation/comprehensive_uncertainty_eval.py --model SpatialPhysicsCNNLSTM --checkpoint models/convection_bioheat_model.pth --samples 1

# 6. Metabolic Bioheat
python evaluation/comprehensive_uncertainty_eval.py --model SpatialPhysicsCNNLSTM --checkpoint models/metabolic_bioheat_model.pth --samples 1
```

### B. Spatial Models (Deterministic)

```bash
# 7. Simple ResNet
python evaluation/comprehensive_uncertainty_eval.py --model SimpleResNet --checkpoint models/simple_resnet_model.pth --samples 1

# 8. Spatial Bioheat
python evaluation/comprehensive_uncertainty_eval.py --model SpatialResNet --checkpoint models/spatial_bioheat_model.pth --samples 1

# 9. Spatial Convection
python evaluation/comprehensive_uncertainty_eval.py --model SpatialResNet --checkpoint models/spatial_convection_model.pth --samples 1

# 10. Spatial Metabolic
python evaluation/comprehensive_uncertainty_eval.py --model SpatialResNet --checkpoint models/spatial_metabolic_model.pth --samples 1
```

### C. Uncertainty Models (Probabilistic)
*These will report NLL, PICP, and MPIW.*

```bash
# 11. Bayesian Head (ResNet)
python evaluation/comprehensive_uncertainty_eval.py --model BayesianResNet --checkpoint checkpoints/bayesian_resnet_head.pth --samples 50

# 12. Full Bayesian ResNet
python evaluation/comprehensive_uncertainty_eval.py --model FullBayesianResNet --checkpoint checkpoints/full_bayesian_resnet.pth --samples 50

# 13. Bayesian PINN
python evaluation/comprehensive_uncertainty_eval.py --model BayesianResNet --checkpoint models/bayesian_pinn.pth --samples 50

# 14. Bayesian CNNLSTM
python evaluation/comprehensive_uncertainty_eval.py --model BayesianCNNLSTM --checkpoint models/bayesian_cnnlstm.pth --samples 50

# 15. Bayesian Metabolic PINN
python evaluation/comprehensive_uncertainty_eval.py --model BayesianCNNLSTM --checkpoint models/bayesian_metabolic_pinn.pth --samples 50

# 16. Bayesian Spatial Convection
python evaluation/comprehensive_uncertainty_eval.py --model BayesianSpatialResNet --checkpoint models/bayesian_spatial_convection.pth --samples 50
```

### D. Ensemble Evaluation
*Note: Requires custom script to aggregate predictions from `checkpoints/ensemble/model_*.pth`.*

```bash
# 17. Ensemble (SimpleResNet)
python evaluation/comprehensive_uncertainty_eval.py --model SimpleResNet --ensemble_dir checkpoints/ensemble --samples 1
```
