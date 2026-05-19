# Makefile for Video Regression Benchmarks

PYTHON := .venv/bin/python
LOG_DIR := logs
MODELS_DIR := models
CHECKPOINTS_DIR := checkpoints

# Ensure directories exist
$(shell mkdir -p $(LOG_DIR) $(MODELS_DIR) $(CHECKPOINTS_DIR))

# Targets
.PHONY: all cnnlstm pretrained_cnnlstm simple_resnet physics_cnnlstm ensemble bayesian full_bayesian spatial_bioheat spatial_convection spatial_metabolic \
        unet_sparse_noprior unet_sparse_withprior tmux_part2 evaluation \
        loso_all loso_temporal loso_spatial loso_uncertainty monitor visualize_loso

# --- LOSO Cross-Validation ---

loso_all:
	@echo "Starting LOSO for all models in parallel..."
	bash scripts/run_loso_benchmarks.sh

loso_temporal:
	@for model in CNNLSTM PretrainedCNNLSTM PhysicsCNNLSTM ConvectionBioheat ConvLTC; do \
		bash scripts/run_loso_benchmarks.sh $$model; \
	done

loso_spatial:
	@for model in SimpleResNet SpatialResNet; do \
		bash scripts/run_loso_benchmarks.sh $$model; \
	done

loso_uncertainty:
	@for model in BayesianResNet FullBayesianResNet BayesianCNNLSTM; do \
		bash scripts/run_loso_benchmarks.sh $$model; \
	done

monitor:
	@bash scripts/monitor_benchmarks.sh

visualize_loso:
	@echo "Generating LOSO Visualization plots..."
	$(PYTHON) evaluation/visualize_loso.py

# --- General ---

all: cnnlstm pretrained_cnnlstm simple_resnet physics_cnnlstm ensemble bayesian full_bayesian spatial_bioheat spatial_convection spatial_metabolic

# --- Evaluation ---

evaluation:
	@echo "Starting Unified Evaluation..."
	$(PYTHON) evaluation/run_unified_evaluation.py --samples 20 2>&1 | tee $(LOG_DIR)/evaluation.log

# --- Part 2: Dense Map Estimation ---

unet_sparse_noprior:
	@echo "Starting U-Net Baseline (No Prior)..."
	$(PYTHON) training/train_unet_sparse.py --run_name unet_sparse_noprior --epochs 50 --no_physics_prior 2>&1 | tee $(LOG_DIR)/unet_sparse_noprior.log

unet_sparse_withprior:
	@echo "Starting U-Net with Physics Prior..."
	$(PYTHON) training/train_unet_sparse.py --run_name unet_sparse_withprior --epochs 50 2>&1 | tee $(LOG_DIR)/unet_sparse_withprior.log

tmux_part2:
	@echo "Launching Part 2 benchmarks in tmux..."
	./launch_benchmarks_tmux.sh part2_baseline

# Target to retrain all models on the new dataset (Unmasked)
retrain_all:
	@echo "Starting Full Retraining (Unmasked) on new data..."
	@mkdir -p $(LOG_DIR)/retrain
	$(PYTHON) training/train_all_models.py --models all --epochs 50 --patience 10 2>&1 | tee $(LOG_DIR)/retrain/all_models_unmasked.log

# Target to retrain all models on the new dataset (Masked)
retrain_all_masked:
	@echo "Starting Full Retraining (Masked) on new data..."
	@mkdir -p $(LOG_DIR)/retrain
	$(PYTHON) training/train_all_models.py --models all --masked --epochs 50 --patience 10 2>&1 | tee $(LOG_DIR)/retrain/all_models_masked.log

# Target to archive old models and start fresh retraining
fresh_start:
	@echo "Archiving legacy models..."
	@mkdir -p models/archive_legacy_data/masked
	@-mv models/*.pth models/archive_legacy_data/ 2>/dev/null || true
	@-mv models/masked/*.pth models/archive_legacy_data/masked/ 2>/dev/null || true
	@echo "Starting parallel retraining streams..."
	@make retrain_all > $(LOG_DIR)/retrain/main_stream.log 2>&1 &
	@make retrain_all_masked > $(LOG_DIR)/retrain/masked_stream.log 2>&1 &
	@echo "Launched background training. Check logs in $(LOG_DIR)/retrain/"

# --- Part 1 Targets ---

# 1. CNNLSTM
cnnlstm: $(MODELS_DIR)/cnnlstm_model.pth

$(MODELS_DIR)/cnnlstm_model.pth:
	@echo "Starting CNNLSTM Training..."
	$(PYTHON) training/train_all_models.py --models cnnlstm --epochs 50 --patience 10 2>&1 | tee $(LOG_DIR)/cnnlstm.log

# 2. Pretrained CNNLSTM
pretrained_cnnlstm: $(MODELS_DIR)/pretrained_cnnlstm_model.pth

$(MODELS_DIR)/pretrained_cnnlstm_model.pth:
	@echo "Starting Pretrained CNNLSTM Training..."
	$(PYTHON) training/train_all_models.py --models pretrained_cnnlstm --epochs 50 --patience 10 2>&1 | tee $(LOG_DIR)/pretrained_cnnlstm.log

# 3. Simple ResNet
simple_resnet: $(MODELS_DIR)/simple_resnet_model.pth

$(MODELS_DIR)/simple_resnet_model.pth:
	@echo "Starting Simple ResNet Training..."
	$(PYTHON) training/train_all_models.py --models simple_resnet --epochs 50 --patience 10 2>&1 | tee $(LOG_DIR)/simple_resnet.log

# 4. Physics CNNLSTM
physics_cnnlstm: $(MODELS_DIR)/physics_cnnlstm_model.pth

$(MODELS_DIR)/physics_cnnlstm_model.pth:
	@echo "Starting Physics CNNLSTM Training..."
	$(PYTHON) training/train_all_models.py --models physics_cnnlstm --epochs 50 --patience 10 2>&1 | tee $(LOG_DIR)/physics_cnnlstm.log

# 5. Ensemble
ensemble: $(CHECKPOINTS_DIR)/ensemble/model_0.pth

$(CHECKPOINTS_DIR)/ensemble/model_0.pth:
	@echo "Starting Ensemble Training..."
	$(PYTHON) training/train_uncertainty.py --mode ensemble --epochs 20 2>&1 | tee $(LOG_DIR)/ensemble.log

# 6. Bayesian Head
bayesian: $(CHECKPOINTS_DIR)/bayesian_resnet.pth

$(CHECKPOINTS_DIR)/bayesian_resnet.pth:
	@echo "Starting Bayesian Head Training..."
	$(PYTHON) training/train_uncertainty.py --mode bayesian --epochs 30 2>&1 | tee $(LOG_DIR)/bayesian_head.log

# 7. Full Bayesian
full_bayesian: $(CHECKPOINTS_DIR)/full_bayesian_resnet.pth

$(CHECKPOINTS_DIR)/full_bayesian_resnet.pth:
	@echo "Starting Full Bayesian Training..."
	$(PYTHON) training/train_uncertainty.py --mode full_bayesian --epochs 30 2>&1 | tee $(LOG_DIR)/full_bayesian.log

# 8. Spatial Bioheat
spatial_bioheat: $(MODELS_DIR)/spatial_bioheat_resnet.pth

$(MODELS_DIR)/spatial_bioheat_resnet.pth:
	@echo "Starting Spatial Bioheat ResNet Training..."
	$(PYTHON) training/train_spatial_bioheat.py --epochs 50 2>&1 | tee $(LOG_DIR)/spatial_bioheat_resnet.log

# 9. Spatial Convection
spatial_convection: $(MODELS_DIR)/spatial_convection_bioheat_resnet.pth

$(MODELS_DIR)/spatial_convection_bioheat_resnet.pth:
	@echo "Starting Spatial Convection Bioheat Training..."
	$(PYTHON) training/train_spatial_convection_bioheat.py --epochs 50 2>&1 | tee $(LOG_DIR)/spatial_convection_bioheat.log

# 10. Spatial Metabolic
spatial_metabolic: $(MODELS_DIR)/spatial_metabolic_bioheat_resnet.pth

$(MODELS_DIR)/spatial_metabolic_bioheat_resnet.pth:
	@echo "Starting Spatial Metabolic Bioheat Training..."
	$(PYTHON) training/train_spatial_metabolic_bioheat.py --epochs 50 2>&1 | tee $(LOG_DIR)/spatial_metabolic_bioheat.log

clean:
	rm -f $(LOG_DIR)/*.log

start_benchmarks:
	@echo "Starting ALL benchmarks (Scalar, Spatial, U-Net, Bayesian PINN) in tmux..."
	@bash scripts/run_benchmarks.sh

# ========================================================================
# AUDIT-DRIVEN TARGETS (added by academic-rigor pass)
# ========================================================================

RESULTS_DIR := results
VALIDATION_DIR := validation_results
DATA_DIR := data/level1_cropped

.PHONY: missing list_missing loso_aggregate master_table validation \
        calibration stat_tests physics_losses_ablation \
        lint format test ci_local docs_pdf consolidate_paper \
        eval_all eval_uq help_audit

## list_missing : Print which model checkpoints are not on disk.
list_missing:
	@$(PYTHON) scripts/find_missing_experiments.py

## missing     : Train only the models that have no checkpoint yet.
missing: list_missing
	@for m in $$($(PYTHON) scripts/find_missing_experiments.py --list-only); do \
	    echo "=== Training missing: $$m ==="; \
	    case $$m in \
	      CNNLSTM)              $(MAKE) cnnlstm ;; \
	      PretrainedCNNLSTM)    $(MAKE) pretrained_cnnlstm ;; \
	      SimpleResNet)         $(MAKE) simple_resnet ;; \
	      PhysicsCNNLSTM)       $(MAKE) physics_cnnlstm ;; \
	      Ensemble)             $(MAKE) ensemble ;; \
	      BayesianResNet)       $(MAKE) bayesian ;; \
	      FullBayesianResNet)   $(MAKE) full_bayesian ;; \
	      SpatialBioheat)       $(MAKE) spatial_bioheat ;; \
	      SpatialConvection)    $(MAKE) spatial_convection ;; \
	      SpatialMetabolic)     $(MAKE) spatial_metabolic ;; \
	      *) echo "No Makefile rule for $$m, skipping." ;; \
	    esac; \
	done

## eval_all    : Run scalar evaluation across every checkpoint we have.
eval_all:
	$(PYTHON) evaluation/evaluate_models.py 2>&1 | tee $(LOG_DIR)/eval_all.log

## eval_uq     : Run uncertainty-aware evaluation across UQ checkpoints.
eval_uq:
	$(PYTHON) evaluation/comprehensive_uncertainty_eval.py 2>&1 | tee $(LOG_DIR)/eval_uq.log

## loso_aggregate : Aggregate per-fold LOSO CSVs into summary statistics.
loso_aggregate:
	$(PYTHON) evaluation/aggregate_loso.py \
	    --pattern '$(RESULTS_DIR)/loso_*_masked.csv' \
	    --pattern '$(RESULTS_DIR)/loso_*_unmasked.csv' \
	    --out-summary $(RESULTS_DIR)/loso_summary.csv \
	    --out-long $(RESULTS_DIR)/loso_per_fold.csv \
	    --allow-empty

## master_table : Build the master CSV + LaTeX table for the paper.
master_table:
	$(PYTHON) evaluation/build_master_table.py \
	    --pattern '$(RESULTS_DIR)/model_*.csv' \
	    --pattern '$(RESULTS_DIR)/tables/*.csv' \
	    --pattern '$(RESULTS_DIR)/loso_summary.csv' \
	    --out-csv $(RESULTS_DIR)/MASTER_RESULTS.csv \
	    --out-tex $(RESULTS_DIR)/MASTER_RESULTS.tex \
	    --allow-empty

## loso_plots  : Generate LOSO figures (MAE bars, field MAE bars, per-fold heatmap, KAN comparison) + LaTeX table.
loso_plots: loso_aggregate
	$(PYTHON) evaluation/plot_loso_results.py \
	    --per-fold $(RESULTS_DIR)/loso_per_fold.csv \
	    --summary $(RESULTS_DIR)/loso_summary.csv \
	    --figdir paper/figures \
	    --tabledir paper/tables

## stat_tests  : Pairwise Wilcoxon p-value matrix across LOSO folds.
stat_tests: loso_aggregate
	$(PYTHON) evaluation/pairwise_stat_tests.py \
	    --input $(RESULTS_DIR)/loso_per_fold.csv \
	    --metric mae \
	    --out-csv $(RESULTS_DIR)/pairwise_wilcoxon.csv \
	    --out-png $(RESULTS_DIR)/pairwise_wilcoxon.png \
	    --allow-empty

## calibration : Reliability + sharpness diagnostics on UQ outputs.
##   Override INPUT and NAME, e.g.: make calibration INPUT=results/bayes_resnet_preds.csv NAME=bayes_resnet
INPUT ?= $(RESULTS_DIR)/uq_predictions.csv
NAME ?= model
calibration:
	$(PYTHON) evaluation/calibration_diagnostics.py \
	    --input $(INPUT) --name $(NAME) --out-dir $(RESULTS_DIR)/calibration

## validation  : Cut-phantom + CEM43 + IoU/Dice end-to-end.
##   Required: MODEL=<registry-name>  CKPT=<path/to/ckpt.pth>
MODEL ?= stub
CKPT ?=
CEM43_THRESH ?= 240
MC ?= 1
validation:
	$(PYTHON) validation/validate_ablation_zone.py \
	    --video_dir data/new_data/videos \
	    --phantom_dir data/new_data/phantoms \
	    --model $(MODEL) \
	    $(if $(CKPT),--checkpoint $(CKPT)) \
	    --cem43_thresh $(CEM43_THRESH) \
	    --mc_samples $(MC) \
	    --output_dir $(VALIDATION_DIR)
	$(PYTHON) validation/analyze_validation_metrics.py \
	    --input $(VALIDATION_DIR)/metrics.csv \
	    --output_dir $(VALIDATION_DIR)

## physics_losses_ablation : Sweep over the new additional physics losses.
physics_losses_ablation:
	@for w in 0.0 0.01 0.1 1.0; do \
	    echo "=== Ablation w_energy=$$w ==="; \
	    $(PYTHON) training/train_spatial_bioheat.py \
	        --epochs 20 --extra-loss energy --extra-weight $$w \
	        --tag energy_w$$w 2>&1 | tee $(LOG_DIR)/ablation_energy_$$w.log; \
	done

## consolidate_paper : Pull paper-related files from sibling branches.
consolidate_paper:
	bash scripts/consolidate_paper_branches.sh

## lint        : Run Black + isort + Flake8 in check mode.
lint:
	black --check --line-length=88 .
	isort --check-only --profile=black --line-length=88 .
	flake8 --max-line-length=88 --extend-ignore=E203,W503,E501 .

## format      : Apply Black + isort.
format:
	black --line-length=88 .
	isort --profile=black --line-length=88 .

## test        : Run unit tests.
test:
	$(PYTHON) -m pytest -q tests

## ci_local    : Lint + tests, the same set CI runs.
ci_local: lint test

## docs_pdf    : Build the paper PDF.
docs_pdf:
	$(MAKE) -C paper

## help_audit  : List the academic-audit targets with their docstrings.
help_audit:
	@grep -E '^## ' Makefile | sed 's/^## //'
