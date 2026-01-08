# Next Actions Checklist

## Immediate Tasks (Monitoring & Validation)
- [x] **Launch Part 1 Retraining:**
    *   `physics_cnnlstm` and `pretrained_cnnlstm` are running.
- [x] **Launch Part 2 Benchmarks:**
    *   `unet_sparse_no/prior` and `hybrid` are running.
- [x] **Launch Part 3 Benchmarks:**
    *   `ltc_unet` and `conv_ltc` are running.
- [ ] **Monitor Convergence:**
    *   Check logs/tmux sessions daily.
    *   Look for instability in the PDE loss term.
- [ ] **Compare Results:**
    *   Once training finishes, generate comparison tables for Part 2/3.
    *   Visualize output heatmaps (using `demo/live_inference.py` or similar).

## Completed Features (Implementation)
- [x] **Implement Hybrid Loss:** Added the Bioheat PDE loss term to `physics/hybrid_loss.py`.
- [x] **Integration:** Created `training/train_unet_hybrid.py` using `BioheatHybridLoss`.
- [x] **LTC Integration:** Implemented `models/latent_ltc.py` and connected it to the U-Net bottleneck.

## Documentation
- [x] **Update Papers:** Added LTC justification to `ltc_section.tex`.
- [ ] **Final Report:** Synthesize all results into `evaluation_report.md` when benchmarks complete.
