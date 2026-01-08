# Makefile for Video Regression Benchmarks

PYTHON := python3
LOG_DIR := logs
MODELS_DIR := models
CHECKPOINTS_DIR := checkpoints

# Ensure directories exist
$(shell mkdir -p $(LOG_DIR) $(MODELS_DIR) $(CHECKPOINTS_DIR))

# Targets
.PHONY: all cnnlstm pretrained_cnnlstm simple_resnet physics_cnnlstm ensemble bayesian full_bayesian spatial_bioheat spatial_convection spatial_metabolic \
        unet_sparse_noprior unet_sparse_withprior tmux_part2

all: cnnlstm pretrained_cnnlstm simple_resnet physics_cnnlstm ensemble bayesian full_bayesian spatial_bioheat spatial_convection spatial_metabolic

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
