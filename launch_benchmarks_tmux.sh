#!/bin/bash

SESSION_NAME="video_regression_benchmarks"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if tmux session exists
if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux new-session -d -s $SESSION_NAME -n "monitor"
    tmux send-keys -t $SESSION_NAME:monitor "htop" C-m
else
    echo "Session $SESSION_NAME already exists. Adding windows..."
fi

launch_benchmark() {
    local name=$1
    local cmd=$2
    
    # Create new window
    tmux new-window -t $SESSION_NAME -n "$name"
    
    # Activate virtual environment if it exists
    tmux send-keys -t $SESSION_NAME:"$name" "source .venv/bin/activate" C-m
    
    # Run command
    echo "Launching $name: $cmd"
    tmux send-keys -t $SESSION_NAME:"$name" "$cmd" C-m
}

# Parse arguments
if [ "$1" == "part2_baseline" ]; then
    launch_benchmark "unet_noprior" "python3 training/train_unet_sparse.py --run_name unet_sparse_noprior --epochs 50 --no_physics_prior"
    launch_benchmark "unet_prior" "python3 training/train_unet_sparse.py --run_name unet_sparse_withprior --epochs 50"
elif [ "$1" == "part2_hybrid" ]; then
    launch_benchmark "unet_hybrid" "python3 training/train_unet_hybrid.py --run_name unet_hybrid_physics --epochs 50 --lambda_physics 0.001"
elif [ "$1" == "part1_all" ]; then
    # Legacy targets from Part 1 if needed
    echo "Part 1 benchmarks not configured in this new script version."
else
    echo "Usage: $0 [part2_baseline]"
    exit 1
fi

echo "Benchmarks launched in tmux session '$SESSION_NAME'"
echo "Attach with: tmux attach -t $SESSION_NAME"
