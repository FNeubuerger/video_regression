# 0003 — uv-managed, lockfile-pinned environment

- **Status:** accepted (2026-11-02)

## Context

Dependencies were a hand-maintained `pip freeze` in `requirements.txt` (89
transitive pins plus unpinned `captum`/`quantus`). The `.venv` on the analysis
host had been copied from the training host: its interpreter symlink resolved
to the local Python 3.10 while the packages were built for 3.12, and every script
shebang pointed at `/mnt/data2/video_regression`, so `import torch` failed and
`make test` could not run. CI installed with `pip install -r requirements.txt || true`
and ran tests with `|| true`, so it never caught a broken environment.

## Decision

- `pyproject.toml` lists only direct dependencies; `[tool.uv] package = false`
  because the repo is a script collection, not an installable package.
- `uv.lock` is committed and is the source of exact versions; environments are
  created with `uv sync --locked` (`make setup`), never copied.
- `torch==2.8.*` / `torchvision==0.23.*` are pinned at the minor level because
  all committed checkpoints were trained with torch 2.8.0; other deps use lower bounds.
- Dependency groups: `dev` (pytest, black/isort/flake8 at the pre-commit versions,
  pre-commit) and `xai` (captum, quantus); both are installed by default.
- `requirements.txt` is generated from the lock (`make lock`) for pip-only
  hosts; CI fails if it drifts from `uv.lock`.
- Python is pinned to 3.12 (`.python-version`, `requires-python = ">=3.12,<3.13"`).

## Consequences

- Fresh resolution moved several transitive packages forward (numpy 2.5, captum
  0.9, wandb 0.31, onnx/onnxruntime). Verified: the full test suite passes,
  LOSO and U-Net checkpoints load with all keys matched, and ONNX export +
  onnxruntime inference work.
- CI now installs the full CUDA torch wheels (several GB) from the lockfile.
