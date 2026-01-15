#!/bin/bash

# Launch Missing Benchmarks (Gaps)
# GPU 0: Dense U-Nets (3 Models)
# GPU 1: Bayesian Physics Scalars (4 Models)

VENV_PATH=".venv/bin/python"
LOG_DIR_UNET="logs/unet"
LOG_DIR_BAYES="logs/physics"
mkdir -p $LOG_DIR_UNET $LOG_DIR_BAYES
EPOCHS=30

echo "Launching Missing Benchmarks..."

# GPU 0: Dense U-Nets
# 1. Standard U-Net
tmux new-session -d -s "unet_std" "CUDA_VISIBLE_DEVICES=0 $VENV_PATH training/train_unet_sparse.py --run_name unet_standard --epochs $EPOCHS --batch_size 16 --no_physics_prior > $LOG_DIR_UNET/unet_standard.log 2>&1"
echo "Launched unet_std (GPU 0)"

# 2. Variational U-Net
tmux new-session -d -s "unet_var" "CUDA_VISIBLE_DEVICES=0 $VENV_PATH training/train_unet_sparse.py --run_name unet_variational --epochs $EPOCHS --batch_size 16 --variational > $LOG_DIR_UNET/unet_variational.log 2>&1"
echo "Launched unet_var (GPU 0)"

# 3. Hybrid U-Net
tmux new-session -d -s "unet_hyb" "CUDA_VISIBLE_DEVICES=0 $VENV_PATH training/train_unet_hybrid.py --run_name unet_hybrid --epochs $EPOCHS --batch_size 16 > $LOG_DIR_UNET/unet_hybrid.log 2>&1"
echo "Launched unet_hyb (GPU 0)"


# GPU 1: Bayesian Physics
# 1. Bayesian PINN
tmux new-session -d -s "bayes_pinn" "CUDA_VISIBLE_DEVICES=1 $VENV_PATH training/train_bayesian_pinn.py --epochs $EPOCHS > $LOG_DIR_BAYES/bayesian_pinn.log 2>&1"
echo "Launched bayes_pinn (GPU 1)"

# 2. Bayesian Convection PINN
tmux new-session -d -s "bayes_conv" "CUDA_VISIBLE_DEVICES=1 $VENV_PATH training/train_bayesian_convection_pinn.py --epochs $EPOCHS > $LOG_DIR_BAYES/bayesian_convection.log 2>&1"
echo "Launched bayes_conv (GPU 1)"

# 3. Bayesian Metabolic PINN
tmux new-session -d -s "bayes_meta" "CUDA_VISIBLE_DEVICES=1 $VENV_PATH training/train_bayesian_metabolic_pinn.py --epochs $EPOCHS > $LOG_DIR_BAYES/bayesian_metabolic.log 2>&1"
echo "Launched bayes_meta (GPU 1)"

# 4. Bayesian Spatial Convection
tmux new-session -d -s "bayes_sp_conv" "CUDA_VISIBLE_DEVICES=1 $VENV_PATH training/train_bayesian_spatial_convection.py --epochs $EPOCHS > $LOG_DIR_BAYES/bayesian_spatial_convection.log 2>&1"
echo "Launched bayes_sp_conv (GPU 1)"

echo "All missing benchmarks launched."
