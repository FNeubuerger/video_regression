#!/bin/bash
# ============================================================================
# Relaunch Failed Benchmarks
# ============================================================================

set -e

WORKSPACE="/mnt/data2/video_regression"
SESSION="video_regression_recovery"

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null || true

echo "Launching Recovery Benchmarks in tmux session: $SESSION"

# 1. Physics CNNLSTM (Failed previously)
tmux new-session -d -s $SESSION -n "physics"
tmux send-keys -t $SESSION:physics "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:physics "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:physics "make physics_cnnlstm" C-m

# 2. Pretrained CNNLSTM (Failed previously)
tmux new-window -t $SESSION -n "pretrained"
tmux send-keys -t $SESSION:pretrained "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:pretrained "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:pretrained "make pretrained_cnnlstm" C-m

# 3. Ensemble (Stalled)
tmux new-window -t $SESSION -n "ensemble"
tmux send-keys -t $SESSION:ensemble "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:ensemble "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:ensemble "make ensemble" C-m

echo "Launched!"
