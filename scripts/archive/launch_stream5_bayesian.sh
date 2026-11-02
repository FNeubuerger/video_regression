#!/bin/bash
# scripts/launch_stream5_bayesian.sh
# STREAM 5: Bayesian Physics Models
# Launches Bayesian PINN variants in a Tmux session

SESSION_NAME="benchmark_stream5_bayesian"
WORKSPACE_DIR="/mnt/data2/video_regression"
VENV_PATH="$WORKSPACE_DIR/.venv"

# Ensure log directory exists
mkdir -p "$WORKSPACE_DIR/logs/part3"

# Check if session exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Killing existing session $SESSION_NAME..."
    tmux kill-session -t $SESSION_NAME
fi

# Create new session (detached) with first window named 'monitor'
tmux new-session -d -s $SESSION_NAME -n "monitor"
tmux send-keys -t $SESSION_NAME:monitor "cd $WORKSPACE_DIR" C-m
tmux send-keys -t $SESSION_NAME:monitor "htop" C-m

# Helper function to create window and run command
launch_window() {
    local name=$1
    local cmd=$2
    local logfile=$3
    
    tmux new-window -t $SESSION_NAME -n "$name"
    tmux send-keys -t $SESSION_NAME:"$name" "cd $WORKSPACE_DIR" C-m
    tmux send-keys -t $SESSION_NAME:"$name" "source $VENV_PATH/bin/activate" C-m
    tmux send-keys -t $SESSION_NAME:"$name" "echo 'Starting $name training...'" C-m
    # Run python unbuffered (-u) and pipe to tee
    tmux send-keys -t $SESSION_NAME:"$name" "python -u $cmd 2>&1 | tee $logfile" C-m
}

# 1. Bayesian PINN
launch_window "bayesian_pinn" \
    "training/train_bayesian_pinn.py --epochs 50 --batch_size 16 --kl 0.1" \
    "logs/part3/bayesian_pinn.log"

# 2. Bayesian Convection PINN
launch_window "bayesian_conv" \
    "training/train_bayesian_convection_pinn.py --epochs 50 --batch_size 16 --kl 0.1" \
    "logs/part3/bayesian_convection_pinn.log"

# 3. Bayesian Metabolic PINN
launch_window "bayesian_meta" \
    "training/train_bayesian_metabolic_pinn.py --epochs 50 --batch_size 16 --kl 0.1" \
    "logs/part3/bayesian_metabolic_pinn.log"

# 4. Bayesian Spatial Convection
launch_window "bayesian_spatial" \
    "training/train_bayesian_spatial_convection.py --epochs 50 --batch_size 16 --kl 0.1" \
    "logs/part3/bayesian_spatial_convection.log"

echo "Launched Stream 5 (Bayesian) in tmux session: $SESSION_NAME"
echo "To view: tmux attach -t $SESSION_NAME"
