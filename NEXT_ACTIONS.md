# Next Actions Checklist

## Immediate Tasks (Phase 2 Validation)
- [x] **Run Baseline Training (No Prior):**
    *   Launched in tmux session `video_regression_benchmarks`.
- [x] **Run Residual Training (With Prior):**
    *   Launched in tmux session `video_regression_benchmarks`.
- [ ] **Compare Results:**
    *   Check validation loss curves in WandB.
    *   Visualize output heatmaps (using `demo/live_inference.py` or similar).

## Upcoming Features (Physics Integration)
- [x] **Implement Hybrid Loss:** Added the Bioheat PDE loss term to `physics/hybrid_loss.py`.
- [x] **Integration:** Created `training/train_unet_hybrid.py` using `BioheatHybridLoss`.
- [x] **Launch Benchmark:** Launched `unet_hybrid_physics` in tmux.
- [ ] **Tune Lambda:** Experiment with weights for the physics loss (currently 0.001).

## Phase 3 (Time & Dynamics)
- [ ] **LTC Integration:** Begin integrating Liquid Time Constant units into the U-Net bottleneck (as researched in `docs/RESEARCH_PART2.md`).
