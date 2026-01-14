# Next Actions Checklist

## Immediate Tasks (Monitoring & Validation)
- [x] **Launch LOSO Benchmarks:** (In progress: `loso_baseline`, `loso_physics`)
- [x] **Inverse Physics Implementation:** (Done: spatial map prediction)
- [ ] **Monitor Convergence:**
    *   Check `loso_physics` loss trends.
    *   Verify that $\alpha$ and $\beta$ maps are not collapsing to uniform constants.
- [ ] **Physics Map Visualization:**
    *   Run `python evaluation/visualize_physics_maps.py` once checkpoints are available.
- [ ] **Advection Analysis:**
    *   Run `python evaluation/visualize_advection.py` to study $\mathbf{v} \cdot \nabla T$ term.

## Completed Features (Implementation)
- [x] **Spatial Physics Prediction:** Added learnable perfusion ($\alpha$) and conductivity ($\beta$) heads.
- [x] **Advection Integration:** Dense Optical Flow integrated into Bioheat loss.
- [x] **LOSO Automation:** Created `scripts/run_loso_benchmarks.sh`.

## Global Strategy: Generalization Validation
- [/] **LOSO Cross-Validation:** (ACTIVE)
    *   Rerun full training/evaluation pipeline for all models using Leave-One-Sequence-Out.
    *   Quantify performance gap between "Random Split" (baseline) and "LOSO" (true generalization).
    *   Store both results set in `results/` for the final paper.

## Documentation
- [x] **Update Papers:** Added LTC justification to `ltc_section.tex`.
- [ ] **Final Report:** Synthesize all results into `evaluation_report.md` when benchmarks complete.
