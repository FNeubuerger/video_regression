#!/bin/bash

SESSION="video_regression_benchmarks"
WORKSPACE="/mnt/data2/video_regression"
LOG_DIR="$WORKSPACE/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Attaching to existing session: $SESSION"
    tmux new-window -t $SESSION -n "spatial_bioheat"
else
    echo "Creating new session: $SESSION"
    tmux new-session -d -s "$SESSION" -n "spatial_bioheat"
fi

tmux send-keys -t $SESSION:spatial_bioheat "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:spatial_bioheat "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:spatial_bioheat "echo 'Starting Spatial Bioheat ResNet Training...'" C-m
tmux send-keys -t $SESSION:spatial_bioheat "python training/train_spatial_bioheat.py --epochs 50 2>&1 | tee $LOG_DIR/spatial_bioheat_resnet.log" C-m

echo "Spatial Bioheat benchmark launched in window 'spatial_bioheat'."
