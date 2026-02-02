#!/bin/bash
# scripts/restart_failed_due_to_dataset.sh
# Restarts models that crashed due to "resize storage" collate error (Dataset issue)

SESSION_NAME="benchmark_retrain"
WORKSPACE_DIR="/mnt/data2/video_regression"
VENV_PATH="$WORKSPACE_DIR/.venv"
LOG_DIR="$WORKSPACE_DIR/logs/retrain"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Check if session exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Killing existing session $SESSION_NAME..."
    tmux kill-session -t $SESSION_NAME
fi

# Create new session (detached) with first window named 'monitor'
tmux new-session -d -s $SESSION_NAME -n "monitor"
tmux send-keys -t $SESSION_NAME:monitor "cd $WORKSPACE_DIR" C-m
tmux send-keys -t $SESSION_NAME:monitor "watch -n 5 'tail -n 5 $LOG_DIR/*.log'" C-m

# Helper function to create window and run command
launch_window() {
    local name=$1
    local cmd=$2
    local logfile=$3
    
    tmux new-window -t $SESSION_NAME -n "$name"
    tmux send-keys -t $SESSION_NAME:"$name" "cd $WORKSPACE_DIR" C-m
    tmux send-keys -t $SESSION_NAME:"$name" "source $VENV_PATH/bin/activate" C-m
    tmux send-keys -t $SESSION_NAME:"$name" "echo 'Starting $name training...'" C-m
    # Run python unbuffered (-u) and pipe to tee/file
    tmux send-keys -t $SESSION_NAME:"$name" "python -u $cmd > $logfile 2>&1" C-m
}

echo "Launching Retrain Session: $SESSION_NAME"

# 1. Bayesian CNNLSTM
launch_window "bayesian_cnnlstm" \
    "training/train_bayesian_cnnlstm.py --epochs 50" \
    "$LOG_DIR/bayesian_cnnlstm.log"

# 2. Bayesian PINN
launch_window "bayesian_pinn" \
    "training/train_bayesian_pinn.py --epochs 50 --batch_size 16 --kl 0.1" \
    "$LOG_DIR/bayesian_pinn.log"

# 3. Bayesian Convection PINN
launch_window "bayesian_convection" \
    "training/train_bayesian_convection_pinn.py --epochs 50 --batch_size 16 --kl 0.1" \
    "$LOG_DIR/bayesian_convection.log"

# 4. Bayesian Metabolic PINN
launch_window "bayesian_metabolic" \
    "training/train_bayesian_metabolic_pinn.py --epochs 50 --batch_size 16 --kl 0.1" \
    "$LOG_DIR/bayesian_metabolic.log"

# 5. Convection Bioheat (Scalar) - Note: 30 epochs for physics scalar
launch_window "convection_bioheat" \
    "training/train_convection_bioheat.py --epochs 30" \
    "$LOG_DIR/convection_bioheat.log"

# 6. Bayesian Spatial Convection (Also crashed, checking if restart helps)
launch_window "bayesian_spatial" \
    "training/train_bayesian_spatial_convection.py --epochs 50 --batch_size 16 --kl 0.1" \
    "$LOG_DIR/bayesian_spatial_convection.log"

echo "All jobs submitted to tmux session '$SESSION_NAME'."
echo "To monitor: tmux attach -t $SESSION_NAME"
echo "Logs in: $LOG_DIR"
