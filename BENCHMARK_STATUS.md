# Benchmark Status Report

**Date:** January 8, 2026
**Status:** Evaluation / Recovery Phase

## Executive Summary
The **Spatial Models** (Bioheat, Convection, Metabolic) have successfully completed training with promising Validation Losses (Best: ~3.15).

However, the **retraining of Physics-based models** and the **Ensemble** training encountered failures. 
- **Physics/Pretrained:** Failed immediately due to environment configuration (`python` command not found).
- **Ensemble:** Stalled/Crashed at Member 4/5.

## Completed Benchmarks (Training)

| Model | Best Validation Loss | Status | Notes |
|-------|----------------------|--------|-------|
| **Simple ResNet** | **2.93** | Complete | Current Baseline SOTA. |
| **Spatial Bioheat** | 3.16 | **Complete** | Training finished successfully (Epoch 50). |
| **Spatial Metabolic** | 3.17 | **Complete** | Training finished successfully (Epoch 50). |
| **Spatial Convection** | 7.37 | **Complete** | Training finished successfully (Epoch 50). Higher loss than others. |
| **Bayesian Head** | 11.06 | Complete | - |
| **CNNLSTM** | 13.67 | Complete | - |
| **Full Bayesian** | 15.06 | Complete | - |

## Failed / In Progress

| Model | Status | Cause/Notes | Action Items |
|-------|--------|-------------|--------------|
| **Physics CNNLSTM** | **Failed** | `python: not found` in Makefile. | **Fix:** Update Makefile to use `python3`. **Retrain.** |
| **Pretrained CNNLSTM** | **Failed** | `python: not found` in Makefile. | **Fix:** Update Makefile to use `python3`. **Retrain.** |
| **Ensemble** | **Stalled** | Stopped at Member 4/5. | Investigate logs/resources. **Restart Member 4 & 5.** |

## Technical Incident Report

### 1. Makefile Environment Error (Active)
*   **Issue:** `Physics CNNLSTM` and `Pretrained CNNLSTM` logs show `/bin/sh: 1: python: not found`.
*   **Diagnosis:** The `Makefile` defines `PYTHON := python`. The current environment (Ubuntu/Venv) likely only creates a `python3` binary/alias, or the shell spawned by `make` does not pick up the venv's `python` alias if it exists.
*   **Resolution:** Change `Makefile` to use `PYTHON := python3` or explicitly point to `.venv/bin/python`.
*   **GitHub Issue:** #1

### 2. Ensemble Training Stall (Active)
*   **Issue:** Ensemble training halted during Member 4 training.
*   **Diagnosis:** Likely resource exhaustion or process interruption. No explicit error in main log, verifying `wandb` run status required.
*   **GitHub Issue:** #2

## Next Steps
1.  **Solve Issue #1:** Update `Makefile` (Done).
2.  **Restart Failures:** Relaunch Physics and Pretrained benchmarks.
3.  **Evaluate:** Run `evaluate_models.py` for the completed Spatial models to populate the `metrics_comparison.csv`.
