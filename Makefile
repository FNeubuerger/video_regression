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
