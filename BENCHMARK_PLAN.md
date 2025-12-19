# Video Temperature Regression: Benchmark & Comparison Plan

This document outlines the suite of 14 active benchmarks and the specific comparative analyses designed to evaluate the contributions of physical constraints, temporal modeling, and uncertainty quantification.

## 1. Active Benchmarks Overview

We are currently training **16 distinct models** across three primary categories.

### A. Temporal Models (Sequence-based)
*Input: Sequence of 5 Frames (RGB + Optical Flow)*
- [x] **CNNLSTM** (Started: 2025-12-19 15:01:44)
- [x] **Pretrained CNNLSTM** (Started: 2025-12-19 15:01:39)
- [x] **Physics CNNLSTM** (Started: 2025-12-19 15:01:40)
- [x] **Bioheat PINN** (Started: 2025-12-19 15:01:49)
- [x] **Convection Bioheat** (Started: 2025-12-19 15:01:48)
- [x] **Metabolic Bioheat** (Started: 2025-12-19 15:01:48)

### B. Spatial Models (Frame-based)
*Input: Single Frame (RGB + Optical Flow)*
- [x] **Simple ResNet** (Started: 2025-12-19 15:01:48)
- [x] **Spatial Bioheat** (Started: 2025-12-19 14:54:33)
- [x] **Spatial Convection** (Started: 2025-12-19 15:00:31)
- [x] **Spatial Metabolic** (Started: 2025-12-19 15:00:47)

### C. Uncertainty Models
- [x] **Ensemble** (Started: 2025-12-19 15:01:42)
- [x] **Bayesian Head** (Started: 2025-12-19 15:01:29)
- [x] **Full Bayesian** (Started: 2025-12-19 15:01:44)
- [x] **Bayesian PINN** (Started: 2025-12-19 15:01:43)
- [x] **Bayesian CNNLSTM** (Started: 2025-12-19 15:01:48)
- [x] **Bayesian Metabolic PINN** (Started: 2025-12-19 14:59:41)
- [x] **Bayesian Spatial Convection** (Started: 2025-12-19 15:10:00)

*(Note: We implemented a Bayesian Spatial ResNet to enable convection physics with uncertainty.)*

---

## 2. Comparative Analysis Plan

We will perform the following pairwise and group comparisons to validate our hypotheses.

### Comparison 1: The "Ladder of Physics" (Temporal)
*Hypothesis: Adding more detailed physical processes improves accuracy and generalization.*

| Model A | Model B | Comparison Goal |
| :--- | :--- | :--- |
| **Pretrained CNNLSTM** | **Bioheat PINN** | Does the basic Bioheat equation (Diff+Perf) outperform pure data-driven learning? |
| **Bioheat PINN** | **Convection Bioheat** | Does explicitly modeling fluid dynamics (blood flow) via Optical Flow help? |
| **Convection Bioheat** | **Metabolic Bioheat** | Does learning a metabolic heat source term ($Q_{met}$) add value? |

### Comparison 2: Spatial vs. Temporal Physics
*Hypothesis: Temporal dynamics ($dT/dt$) are crucial for accurate bioheat modeling, but steady-state spatial constraints still offer regularization benefits.*

| Temporal Model | Spatial Equivalent | Comparison Goal |
| :--- | :--- | :--- |
| **Bioheat PINN** | **Spatial Bioheat** | Value of $dT/dt$ vs. pure Laplacian smoothing. |
| **Convection Bioheat** | **Spatial Convection** | Can we model convection effectively in a steady-state (single frame) regime? |
| **Metabolic Bioheat** | **Spatial Metabolic** | Impact of metabolic terms with vs. without temporal evolution. |

### Comparison 3: Uncertainty Quantification Strategies
*Hypothesis: B-PINN provides the best trade-off between calibration and accuracy.*

| Strategy | Pros | Cons | Metric to Compare |
| :--- | :--- | :--- | :--- |
| **Ensemble** | Robust, simple | High training cost (5x) | NLL, ECE |
| **Bayesian Head** | Fast, lightweight | Ignores feature uncertainty | NLL, Inference Time |
| **Full Bayesian** | Captures all uncertainty | Hard to train, slow inference | NLL, Weight Histograms |
| **Bayesian PINN** | Physically constrained | Complex loss landscape | Physics Residual vs. Uncertainty |

### Comparison 4: The "Ultimate" Showdown
*Identifying the State-of-the-Art (SOTA) for this task.*

*   **Baseline SOTA**: Pretrained CNNLSTM
*   **Physics SOTA**: Metabolic Bioheat (Theoretical best physics)
*   **Uncertainty SOTA**: Bayesian PINN (Theoretical best reliability)

**Key Metrics:**
1.  **MSE/MAE**: Raw accuracy.
2.  **Physics Residual**: How well does it obey the laws of physics?
3.  **Inference Latency**: Can it run on the edge?
4.  **Calibration Error**: Are the confidence intervals reliable?

## 3. Next Steps
1.  Wait for all 50 epochs to complete.
2.  Generate "Physics Compliance Plots": Plot the distribution of the physics loss term for all models on the Test set.
3.  Generate "Uncertainty Calibration Curves": Compare Ensemble vs. Bayesian approaches.
4.  Select the top 3 models for the final paper results table.
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
