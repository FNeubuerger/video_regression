#!/bin/bash
# ============================================================================
# Video Regression Benchmark Launcher
# ============================================================================
# This script launches the training benchmarks in a persistent tmux session.
#
# Usage:
#   ./run_benchmarks.sh
#
# The benchmarks run in: tmux session "video_regression_benchmarks"
# Logs are saved to: logs/
# ============================================================================

set -e

WORKSPACE="/mnt/data2/video_regression"
SESSION="video_regression_benchmarks"
LOG_DIR="$WORKSPACE/logs"

# Create logs directory
mkdir -p $LOG_DIR

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null || true

echo "============================================================================"
echo "Launching Video Regression Benchmarks (FULLY PARALLEL)"
echo "============================================================================"
echo ""
echo "Session:      $SESSION"
echo "Workspace:    $WORKSPACE"
echo "Logs:         $LOG_DIR"
echo "WandB Project: video-temperature-regression"
echo ""
echo "The benchmarks will run in a detached tmux session."
echo "You can safely disconnect - everything will keep running."
echo ""

# Create the session and first window (CNNLSTM)
tmux new-session -d -s $SESSION -n "cnnlstm"

# 1. CNNLSTM
tmux send-keys -t $SESSION:cnnlstm "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:cnnlstm "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:cnnlstm "echo 'Starting CNNLSTM Training...'" C-m
tmux send-keys -t $SESSION:cnnlstm "python training/train_all_models.py --models cnnlstm --epochs 50 --patience 10 2>&1 | tee $LOG_DIR/cnnlstm.log" C-m

# 2. Pretrained CNNLSTM
tmux new-window -t $SESSION -n "pretrained"
tmux send-keys -t $SESSION:pretrained "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:pretrained "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:pretrained "echo 'Starting Pretrained CNNLSTM Training...'" C-m
tmux send-keys -t $SESSION:pretrained "python training/train_all_models.py --models pretrained_cnnlstm --epochs 50 --patience 10 2>&1 | tee $LOG_DIR/pretrained_cnnlstm.log" C-m

# 3. Simple ResNet
tmux new-window -t $SESSION -n "resnet"
tmux send-keys -t $SESSION:resnet "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:resnet "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:resnet "echo 'Starting Simple ResNet Training...'" C-m
tmux send-keys -t $SESSION:resnet "python training/train_all_models.py --models simple_resnet --epochs 50 --patience 10 2>&1 | tee $LOG_DIR/simple_resnet.log" C-m

# 4. Physics CNNLSTM
tmux new-window -t $SESSION -n "physics"
tmux send-keys -t $SESSION:physics "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:physics "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:physics "echo 'Starting Physics CNNLSTM Training...'" C-m
tmux send-keys -t $SESSION:physics "python training/train_all_models.py --models physics_cnnlstm --epochs 50 --patience 10 2>&1 | tee $LOG_DIR/physics_cnnlstm.log" C-m

# 5. Ensemble
tmux new-window -t $SESSION -n "ensemble"
tmux send-keys -t $SESSION:ensemble "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:ensemble "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:ensemble "echo 'Starting Ensemble Training...'" C-m
tmux send-keys -t $SESSION:ensemble "python training/train_uncertainty.py --mode ensemble --epochs 20 2>&1 | tee $LOG_DIR/ensemble.log" C-m

# 6. Bayesian Head
tmux new-window -t $SESSION -n "bayesian_head"
tmux send-keys -t $SESSION:bayesian_head "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:bayesian_head "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:bayesian_head "echo 'Starting Bayesian Head Training...'" C-m
tmux send-keys -t $SESSION:bayesian_head "python training/train_uncertainty.py --mode bayesian --epochs 30 2>&1 | tee $LOG_DIR/bayesian_head.log" C-m

# 7. Full Bayesian
tmux new-window -t $SESSION -n "full_bayesian"
tmux send-keys -t $SESSION:full_bayesian "cd $WORKSPACE" C-m
tmux send-keys -t $SESSION:full_bayesian "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:full_bayesian "echo 'Starting Full Bayesian Training...'" C-m
tmux send-keys -t $SESSION:full_bayesian "python training/train_uncertainty.py --mode full_bayesian --epochs 30 2>&1 | tee $LOG_DIR/full_bayesian.log" C-m

echo "✅ Benchmarks launched in tmux session: $SESSION"
echo ""
echo "============================================================================"
echo "How to Monitor:"
echo "============================================================================"
echo ""
echo "1. Attach to session:"
echo "   tmux attach -t $SESSION"
echo ""
echo "2. Navigate windows:"
echo "   Ctrl+b, then number (0-6) or n/p"
echo ""
echo "3. Detach:"
echo "   Ctrl+b, then d"
echo ""
echo "4. View logs:"
echo "   tail -f logs/*.log"
echo ""
echo "5. Check WandB dashboard:"
echo "   https://wandb.ai/fneubuerger/video-temperature-regression"
echo ""
echo "============================================================================"
