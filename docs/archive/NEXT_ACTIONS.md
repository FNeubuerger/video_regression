# Next Actions Checklist

## Immediate Tasks (Monitoring & Validation)
- [x] **Launch LOSO Benchmarks:** (Active for 8 architectures)
- [x] **Inverse Physics Implementation:** (Done: spatial map prediction)
- [x] **Monitoring Pipeline:** (Done: `make monitor` and `scripts/monitor_loso.sh`)
- [ ] **Monitor Convergence:**
    *   Use `make monitor` to check all 8 LOSO sessions.
    *   Verify that $\alpha$ and $\beta$ maps are not collapsing to uniform constants.
- [ ] **Physics Map Visualization:**
    *   Run `python evaluation/visualize_physics_maps.py` once checkpoints are available.

## Completed Features (Implementation)
- [x] **Spatial Physics Prediction:** Added learnable perfusion ($\alpha$) and conductivity ($\beta$) heads.
- [x] **Advection Integration:** Dense Optical Flow integrated into Bioheat loss.
- [x] **LOSO Automation:** Created `scripts/run_loso_benchmarks.sh`.
- [x] **LOSO Scoped Evaluation:** Updated `generate_tables.py` and `run_unified_evaluation.py`.

## Global Strategy: Generalization Validation
- [x] **LOSO Cross-Validation:** (ACTIVE)
    *   Rerun full training/evaluation pipeline for all models using Leave-One-Sequence-Out.
    *   Quantify performance gap between "Random Split" (baseline) and "LOSO" (true generalization).
    *   Store both results set in `results/` for the final paper.
- [x] **Automated Reporting:** All evaluation steps now integrated into `make evaluation`.
- [x] **Generalization Gap Quantified:** LOSO vs. standard-split comparison documented in `paper/main.tex` ("Cross-Subject Generalization"); root cause traced to unregularized LOSO training.
- [ ] **Overfitting Mitigation (2026-08-26):** `evaluation/loso_cross_validation.py` now supports `--weight_decay`, `--augment`, `--seed`. Regularized retraining of `SimpleResNet`/`CNNLSTM` is running (`scripts/run_loso_benchmarks_regularized.sh`, tmux sessions `loso_reg_simpleresnet`/`loso_reg_cnnlstm`). Next: fill `paper/tables/loso_regularized.tex` with real results and extend to remaining overfitting-prone models (`ConvectionBioheat`, `BayesianResNet`, `KANResNet`).

## Documentation
- [x] **Update Papers:** Added LTC justification to `ltc_section.tex`.
- [ ] **Final Report:** Synthesize all results into `evaluation_report.md` when benchmarks complete.
