# Real-Time Ultrasonic Temperature Monitoring for Tumor Ablation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)

Code for the paper **"Hybrid CNN-LSTM and Physics-Informed Architectures for
Real-Time Tumor Ablation Monitoring on Edge Devices"** (draft in [`paper/`](paper/)).

During microwave ablation (MWA) the tissue temperature must be monitored, but
direct measurement is invasive. This project estimates temperature directly from
ultrasound video (RGB frames + dense optical flow) with lightweight CNN/LSTM,
physics-informed (Pennes bioheat), Bayesian, KAN and liquid-time-constant models,
and benchmarks them for edge deployment (Raspberry Pi 4, Jetson Nano/Orin).

## Status at a glance

- **Current results:** [`docs/RESULTS.md`](docs/RESULTS.md) — LOSO cross-validation
  over 15 phantom sequences. Best LOSO MAE is ~23 K (SpatialResNet, regularized
  SimpleResNet); the large gap to mixed-split numbers is the main open problem.
- **Open work:** [`TODO.md`](TODO.md) (single source of truth for tasks).
- **Paper:** [`paper/main.tex`](paper/main.tex), build with `make docs_pdf`.

## Documentation

| Document | Content |
| :--- | :--- |
| [docs/RESULTS.md](docs/RESULTS.md) | Headline numbers, interpretation, caveats |
| [docs/EXPERIMENTAL_SETUP.md](docs/EXPERIMENTAL_SETUP.md) | Phantom setup, data acquisition |
| [docs/MODEL_ARCHITECTURE_DETAILS.md](docs/MODEL_ARCHITECTURE_DETAILS.md) | Model architectures |
| [docs/METHODOLOGY_PART2.md](docs/METHODOLOGY_PART2.md) | Dense temperature-map estimation |
| [docs/RESEARCH_PART2.md](docs/RESEARCH_PART2.md), [docs/PLAN_CONVLTC.md](docs/PLAN_CONVLTC.md) | LTC / physics-prior research notes |
| [research/XAI_PLAN.md](research/XAI_PLAN.md) | Explainability (Captum/Quantus) plan |
| [docs/decisions/](docs/decisions/) | Architecture/decision records |
| [docs/archive/](docs/archive/) | Superseded status reports and plans (historical) |

## Repository layout

```
models/        Model definitions (backbones, bayesian, kan, conv_ltc, latent_ltc, dense_heads)
physics/       Physics-informed losses (Newton cooling, Pennes bioheat, advection, hybrid)
training/      Training entry points (train_all_models.py + per-model scripts)
evaluation/    LOSO driver, aggregation, statistics, plots, XAI evaluation
benchmarks/    ONNX export + edge-deployment benchmarking
simulation/    Edge hardware profiles and latency emulator
validation/    Ablation-zone (cut-phantom / CEM43 / IoU) validation
demo/          Live inference, clinical demo and XAI dashboard (see demo/README.md)
utils/         Datasets, model registry, augmentation, seeding, ONNX export
scripts/       tmux launchers and monitors (scripts/archive/ = old one-off scripts)
tests/         pytest suite
paper/         LaTeX source, tables and figures
results/       Committed metrics (per-fold LOSO CSVs, summaries, plots)
```

Not tracked in git: `data/`, `checkpoints/`, `logs/`, `wandb/`, `*.pth`, `*.onnx`, `*.mp4`.

## Setup

```bash
git clone git@github.com:FNeubuerger/video_regression.git
cd video_regression
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
pre-commit install          # Black / isort / flake8 hooks
make test                   # pytest
```

Data is expected in `data/level1_cropped/` (15 sequences `US_001` … `US_015`).
Shell launchers default to `<repo>/.venv/bin/python`; override with
`VENV_PATH=/path/to/python`.

## Usage

### Training (standard split)

```bash
python training/train_all_models.py --models cnnlstm --epochs 50 --masked --seed 42
```

`--masked` suppresses thermometer artifacts that otherwise act as a shortcut.

### LOSO cross-validation (primary protocol)

```bash
python evaluation/loso_cross_validation.py --model SimpleResNet --epochs 10 --masked \
    --weight_decay 1e-4 --augment --seed 42
```

Writes `results/loso_<Model>_<tag>.csv` (tag e.g. `masked` or `seed42_aug_masked`)
and a provenance JSON (seed, git sha, hardware) next to every fold checkpoint.

| Goal | Command |
| :--- | :--- |
| All models (tmux, 2 GPUs) | `make loso_all` |
| One model | `make model-CNNLSTM` |
| Model groups | `make loso_temporal` / `loso_spatial` / `loso_uncertainty` |
| Regularized rerun | `make loso_regularized [MODEL_NAME=SimpleResNet]` |
| Monitor | `make monitor` or `bash scripts/monitor_loso.sh` |
| Aggregate + stats + plots | `make loso_aggregate stat_tests loso_plots` |

### Evaluation

```bash
make evaluation     # unified scalar + uncertainty evaluation
make master_table   # results/MASTER_RESULTS.{csv,tex}
make help_audit     # list all audit/evaluation targets
```

### Edge benchmarks

```bash
bash benchmarks/run_edge_benchmarks.sh   # ONNX export + simulated device benchmarks
```

Individual steps: `benchmarks/convert_to_onnx.py`, `benchmarks/benchmark_deployment.py
--frames 50 --onnx`, `benchmarks/analyze_results.py`.

### Demos

See [demo/README.md](demo/README.md) (live inference, clinical demo, XAI dashboard).

## Method summary

- **Input:** sequences of 5 frames × 5 channels (RGB + Farneback optical flow dx, dy), 64×64.
- **Physics-informed loss:** $\mathcal{L} = \mathcal{L}_{MSE} + \lambda \mathcal{L}_{phys}$
  with Newton cooling, Pennes bioheat (perfusion/metabolic terms) and advection
  via optical flow; spatial variants learn perfusion/conductivity maps.
- **Uncertainty:** deep ensembles, Bayesian NNs (`torchbnn`), Bayesian PINNs.

## Citation

```bibtex
@inproceedings{neubuerger2025hybrid,
  title={Hybrid CNN-LSTM and Physics-Informed Architectures for Real-Time Tumor Ablation Monitoring on Edge Devices},
  author={Neubuerger, Felix and Nawrath, Helena},
  year={2025}
}
```

## License

MIT, see [LICENSE](LICENSE).
