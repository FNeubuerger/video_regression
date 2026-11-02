#!/bin/bash
# Relaunch failed benchmarks after code fixes
# Target: Bayesian PINN, Bayesian Convection, Hybrid U-Net

SESSION="video_regression_recovery_part2"
WORKSPACE="/mnt/data2/video_regression"
LOG_PHYS="logs/physics"
LOG_UNET="logs/unet"

mkdir -p $LOG_PHYS
mkdir -p $LOG_UNET

# Kill existing recovery session if it exists to avoid dupes
tmux kill-session -t $SESSION 2>/dev/null || true

echo "Launching Recovery (Part 2) in tmux session: $SESSION"

# 1. Bayesian PINN
tmux new-session -d -s $SESSION -n "bpinn"
tmux send-keys -t $SESSION:bpinn "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:bpinn "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:bpinn "python training/train_bayesian_pinn.py --epochs 30 2>&1 | tee $LOG_PHYS/bayesian_pinn.log" C-m
echo "Restarted Bayesian PINN"

# 2. Bayesian Convection (New Window)
tmux new-window -t $SESSION -n "bconv"
tmux send-keys -t $SESSION:bconv "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:bconv "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:bconv "python training/train_bayesian_convection_pinn.py --epochs 30 2>&1 | tee $LOG_PHYS/bayesian_convection.log" C-m
echo "Restarted Bayesian Convection"

# 3. Hybrid U-Net (New Window)
tmux new-window -t $SESSION -n "uhybrid"
tmux send-keys -t $SESSION:uhybrid "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:uhybrid "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:uhybrid "python training/train_unet_hybrid.py --run_name unet_hybrid_recovered --epochs 30 --lambda_physics 0.001 2>&1 | tee $LOG_UNET/unet_hybrid.log" C-m
echo "Restarted Hybrid U-Net"

echo "All failed streams relaunched! Check monitor."
