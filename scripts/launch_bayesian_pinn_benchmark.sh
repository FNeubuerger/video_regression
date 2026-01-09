#!/bin/bash

SESSION="video_regression_benchmarks"
WORKSPACE="/mnt/data2/video_regression"
LOG_DIR="$WORKSPACE/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Attaching to existing session: $SESSION"
    tmux new-window -t $SESSION -n "bayesian_pinn"
else
    echo "Creating new session: $SESSION"
    tmux new-session -d -s "$SESSION" -n "bayesian_pinn"
fi

tmux send-keys -t $SESSION:bayesian_pinn "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:bayesian_pinn "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:bayesian_pinn "echo 'Starting Bayesian PINN Training...'" C-m
tmux send-keys -t $SESSION:bayesian_pinn "python training/train_bayesian_pinn.py --epochs 50 2>&1 | tee $LOG_DIR/bayesian_pinn.log" C-m

echo "Bayesian PINN benchmark launched in window 'bayesian_pinn'."
