# Academic Audit — Video Regression for MWA Monitoring

**Date:** May 18, 2026
**Scope:** Codebase, benchmarking plan, paper, validation pipeline
**Target Venues:** IEEE TMI, MICCAI, MedIA

---

## Executive Summary

The project has substantial breadth (20+ models, physics-informed variants, UQ, edge deployment, phantom validation) but suffers from **fragmented execution**: incomplete LOSO results, broken evaluation aggregators, reproducibility gaps, and a paper draft that is ~40% incomplete. None of the gaps are conceptual — they are integration, rigor, and finishing work.

**Critical path to submission (≈4 weeks):**
1. Finish LOSO cross-validation and aggregate fold statistics with CIs.
2. Fix shape-mismatch bugs in evaluation aggregators (block final tables).
3. Resolve paper TODOs (phantom specs, antenna, dataset diversity, related work).
4. Add reproducibility scaffolding (seeding, config export, pre-commit, CI).
5. Make validation pipeline scientifically defensible (model loading, GT protocol, threshold sensitivity, registration).
6. Expand bibliography (≥20 refs), write Related Work, and add gap statement.

---

## Area-by-Area Verdicts

| Area | Verdict | Severity | Effort |
|---|---|---|---|
| Paper (main.tex, supplementary) | **WEAK** | High | Medium |
| Plans & Status | Adequate | Medium | Low |
| Models (`models/`) | Adequate | Medium | Low |
| Training (`training/`) | **WEAK** | High | Medium |
| Evaluation (`evaluation/`) | Adequate | Medium | Low (bugfix) |
| Validation (`validation/`) | **WEAK** | High | High |
| Data layout | Adequate | Medium | Low |
| Repro / Infra | **WEAK** | High | Low–Medium |
| Results artifacts | Adequate | Medium | Low |
| Edge benchmarks | **WEAK** | Medium | Medium |
| Research notes / references | Adequate | Medium | Low |

---

## 1. Paper — WEAK

**Findings**
- 4 stated contributions: CNN-LSTM regression, physics-informed variants, edge deployment, UQ.
- 4 unresolved TODO placeholders: phantom material spec, probe frequency/geometry, raw image resolution, dataset diversity.
- Results section, statistical tests, LOSO generalization analysis, and clinical validation are missing or stubbed.
- Only 5 references in `references.bib` (He2016, Farneback2003, Blundell2015, Raissi2019, Lechner2020).
- `supplementary_material.tex` and `ltc_section.tex` exist but PDF build not validated.

**Actions**
- Resolve all `% TODO` placeholders by pulling content from `docs/EXPERIMENTAL_SETUP.md` and `research/clinical_context.md`.
- Integrate completed LOSO numbers into a Results section with mean ± 95% CI.
- Write Related Work (gap statement vs. PINN, MWA monitoring, and ultrasonic thermometry literature).
- Expand `references.bib` to ≥20 entries; cite clinical guidelines on CEM43 and 5–10 mm safety margins.

---

## 2. Plans & Status — Adequate

**Findings**
- `BENCHMARK_PLAN.md` is comprehensive (4 tiers: temporal, spatial, dense, dynamics, UQ).
- Dataset migration to `level1_cropped` (Jan 2026) reset all benchmarks.
- LOSO is mid-flight; several streams crashed (Bayesian + Physics collate error) and were restarted Feb 2, 2026.
- No aggregated LOSO results yet.

**Actions**
- Enforce a fold-level timeout and write per-fold CSVs immediately on completion.
- Auto-aggregate to `results/loso_summary.csv` with mean, std, SE, 95% CI per model.
- Add a single random-split control (e.g., BioheatPINN) to quantify the **generalization gap** vs LOSO.
- Archive legacy/raw-data results separately and report dataset-quality delta.

---

## 3. Models — Adequate

**Findings**
- ~18 model classes implemented across backbones, dense heads, Bayesian, ConvLTC/LatentLTC.
- `models/__init__.py` only exports ~5 classes — many models are not importable through the package surface.
- Legacy classes (`PretrainedCNN`, old `SpatialResNet` variants) coexist with the active ones; naming is inconsistent (e.g., `SpatialPhysicsCNNLSTM`).

**Actions**
- Update `models/__init__.py` to export all production models; move legacy into `models/archive_legacy_data/`.
- Standardize naming: `PhysicsResNet`, `SpatialBioheat`, `SpatialConvectionBioheat`, `SpatialMetabolicBioheat`.
- Make `utils/model_registry.py` the single source of truth: `name -> (class, default hyperparams, ckpt path)`.

---

## 4. Training — WEAK

**Findings**
- 17 model-specific training scripts duplicate the same data loader / loss / optimizer / early-stopping logic.
- No global seeding (`torch.manual_seed`, `np.random.seed`, `random.seed`) detected in master scripts.
- No config or provenance export (seed, commit hash, CUDA / torch versions, data version).
- `--masked` flag semantics are not documented.

**Actions**
- Add `utils/seed_utils.py` with `set_global_seed(seed)`; call at the top of every training entrypoint and persist to checkpoint.
- Save a `run_config.json` per checkpoint with seed, hyperparams, git SHA, torch/CUDA, data version, hostname, GPU model.
- Consolidate into a single `train.py --model NAME --config YAML` entrypoint; keep wrappers only for tmux orchestration.
- Document masking semantics (which channels, which pixels, why).

---

## 5. Evaluation — Adequate (bugs block final tables)

**Findings**
- `run_unified_evaluation.py`, `comprehensive_uncertainty_eval.py`, `loso_cross_validation.py` exist and cover most needs.
- Known shape mismatches: `(10776,) vs (43104,)` in `evaluate_models.py`; `(10776,) vs (21552,)` in `comprehensive_uncertainty_eval.py` — these block `metrics_comparison.csv`.
- ECE / NLL are computed for UQ models, but **reliability diagrams** are missing.
- No statistical significance tests between model pairs.
- Physics-map diagnostics (α, β spatial maps) not consistently saved.

**Actions**
- Trace the frame-vs-sequence reduction; the mismatches look like a sequence-length factor (×2, ×4). Add explicit shape asserts at the start of every evaluator.
- Add reliability diagrams (predicted-vs-empirical bins) and **sharpness** (avg σ) to `generate_scientific_plots.py`.
- Compute paired Wilcoxon / Mann-Whitney across folds for every model pair and emit a p-value matrix.
- Persist α/β maps + Bioheat residual maps per validation sample for qualitative figures.

---

## 6. Validation (CEM43 / Phantom) — WEAK

**Findings**
- `validation/validate_ablation_zone.py` implements CEM43, IoU, Dice and supports multi-view GT, but the model-loading function is a placeholder (linear pixel→°C scaling).
- Ground-truth segmentation is naive intensity thresholding; no morphology, no inter-rater check.
- No registration between the ultrasound ROI (video) and the cut-plane phantom photograph.
- Single CEM43 threshold is hard-coded; no sensitivity analysis or clinical justification.
- No uncertainty propagation from UQ models into the predicted ablation zone.

**Actions**
- Wire the actual trained model into `predict_temperature_maps` (start with BioheatPINN); load via `model_registry`.
- Document and codify the **GT capture protocol**: camera, scale bar, lighting, cutting plane orientation, fiducials.
- Replace threshold segmentation with level-set or GrabCut; have at least one expert annotate a subset for inter-rater (Cohen κ).
- Sweep CEM43 ∈ [30, 240] and report IoU/Dice as a function of threshold; cite literature for canonical thresholds (43°C·240 min for necrosis).
- Propagate model uncertainty: report **probabilistic ablation zone** (P(CEM43 > τ) > 0.5) and IoU with credible intervals.
- For two cut planes, register both with the predicted 3D temperature field (or document the 2D-only limitation explicitly).

---

## 7. Data — Adequate

**Findings**
- Layered data: `level0_raw`, `level1_cropped`, `sequence_1..8`, `new_data`.
- Sensor coordinates stored as JSON (good).
- No `DATA_VERSION.txt`, no per-sequence metadata (agar %, egg-white %, power, duration), no integrity hash, no documented split.

**Actions**
- Emit `data/DATA_MANIFEST.json` per sequence: date, phantom recipe, power W, duration s, frame count, sensor positions, sync error.
- Add MD5 checksums for `level1_cropped`; verify in the data loader on first use.
- Write `data/SPLITS.md` showing the LOSO fold mapping (test sequence per fold).

---

## 8. Reproducibility / Infrastructure — WEAK

**Findings**
- `Makefile` covers training/eval/monitoring targets.
- `requirements.txt` lists ~27 packages but no pin scheme guarantee.
- `tests/` folder exists with several test files but no `pytest.ini`, no CI.
- No `.pre-commit-config.yaml`; Black formatting from `.github/copilot-instructions.md` is **not enforced**.
- No GitHub Actions workflow.

**Actions**
- Add `.pre-commit-config.yaml` (Black, isort, Flake8) and run `pre-commit install`.
- Add `.github/workflows/test.yml`: lint + `pytest -q` on push/PR.
- Add `make lint`, `make format`, `make test` targets.
- Pin core deps (torch, numpy, opencv, wandb) with `==` or compatible-release `~=`.
- Add `config/train_defaults.yaml` consumed by all training scripts.

---

## 9. Results Artifacts — Adequate

**Findings**
- `results/tables/` has temporal/spatial/uncertainty `.tex` and `.csv` outputs.
- `results/plots/` has ~30 PNGs (FPS, efficiency frontier, architecture comparison, physics maps).
- `results/uncertainty_eval/` covers Bayesian/Ensemble.
- LOSO summary is **not present yet**.
- No single master table integrating accuracy + latency + UQ + LOSO gap.

**Actions**
- Generate `results/loso_summary.csv` (model, fold, mae, rmse, nll, ece) and a roll-up table.
- Build `results/MASTER_RESULTS.csv`: model | MAE | RMSE | latency_cpu_ms | latency_onnx_ms | params_M | ECE | NLL | LOSO_gap.
- Wire generation into `make evaluation` so the master table is always reproducible from checkpoints.

---

## 10. Edge Benchmarks — WEAK

**Findings**
- `benchmarks/convert_to_onnx.py`, `benchmarks/benchmark_deployment.py`, `run_edge_benchmarks.sh` exist.
- FPS plots exist for Pi 4 / Jetson Nano / RTX 3090 but no in-repo evidence of actual hardware runs.
- No ONNX numerical-equivalence test; no memory / energy figures.

**Actions**
- Add unit test that asserts PyTorch vs ONNX agreement within 1% on a fixed batch.
- Capture real device measurements where possible; otherwise label simulated results clearly in the paper.
- Add memory footprint, peak RAM, and (where feasible) energy per inference.

---

## 11. Research Notes / Literature — Adequate

**Findings**
- `research/clinical_context.md` has solid MWA background (indications, margins, mechanism).
- `research/literature/reading_notes.csv` indexes ~21 papers.
- `references.bib` only cites 5 papers in `main.tex`.

**Actions**
- Promote ≥15 entries from `reading_notes.csv` to `references.bib`.
- Add recent (2022–2025) PINN, neural ODE, and medical-imaging UQ papers.
- Synthesize a Related Work section with an explicit gap statement: "Unlike X we combine Y with edge deployment and uncertainty-aware ablation zone validation."

---

## Recommended Workstreams (Parallelizable)

**Workstream A — Reproducibility & Infra (1–2 days)**
- Seeding utility, run-config export, pre-commit, CI, Makefile targets, dep pinning.

**Workstream B — Evaluation Hardening (2–3 days)**
- Fix shape mismatches, add reliability diagrams + sharpness, add Wilcoxon matrix, persist physics maps, build master results table.

**Workstream C — LOSO Completion (in flight)**
- Babysit running sessions, write per-fold CSVs, aggregate, archive checkpoints, compute generalization gap vs random split.

**Workstream D — Validation Rigor (3–5 days)**
- Real model loading, GT capture protocol, robust segmentation, threshold sensitivity, uncertainty-aware ablation zone, multi-view handling.

**Workstream E — Paper (1–2 weeks, partially blocked by B/C/D)**
- Resolve TODOs, write Results + Related Work, expand `.bib`, build PDF in CI.

---

## Risks

- **LOSO never finishes for all models** → fall back to a documented subset (e.g., top 6) and explicitly mark deferred ones.
- **Phantom GT protocol underspecified** → reviewers will reject clinical-validation claim; mitigate by framing validation as proof-of-concept on 2D cut planes with explicit limitations.
- **Reproducibility gaps surface in peer review** → preempt by publishing seeds, configs, checksums, and CI logs alongside the paper.

---

## Suggested Next Concrete Steps (this week)

1. Run `make` target survey and fix shape-mismatch bugs in `evaluation/evaluate_models.py` and `evaluation/comprehensive_uncertainty_eval.py`.
2. Add `utils/seed_utils.py` and wire it into all training entrypoints.
3. Add `.pre-commit-config.yaml` and a minimal GitHub Actions workflow.
4. Aggregate whatever LOSO folds have completed into `results/loso_summary.csv` (partial is fine for now).
5. Wire the BioheatPINN checkpoint into `validation/validate_ablation_zone.py`.
6. Open a tracking issue per workstream (A–E) so progress is visible.
