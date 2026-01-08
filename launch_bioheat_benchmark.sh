#!/bin/bash

SESSION="video_regression_benchmarks"
WORKSPACE="/mnt/data2/video_regression"
LOG_DIR="$WORKSPACE/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Attaching to existing session: $SESSION"
    tmux new-window -t $SESSION -n "bioheat"
else
    echo "Creating new session: $SESSION"
    tmux new-session -d -s "$SESSION" -n "bioheat"
fi

tmux send-keys -t $SESSION:bioheat "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:bioheat "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:bioheat "echo 'Starting Advanced Bioheat CNNLSTM Training...'" C-m
tmux send-keys -t $SESSION:bioheat "python training/train_bioheat.py --epochs 50 2>&1 | tee $LOG_DIR/advanced_bioheat_cnnlstm.log" C-m

echo "Advanced Bioheat benchmark launched in window 'bioheat'."
