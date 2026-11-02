#!/bin/bash

SESSION_NAME="video_regression_benchmarks"

# Check if tmux session exists
if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux new-session -d -s $SESSION_NAME -n "monitor"
    tmux send-keys -t $SESSION_NAME:monitor "htop" C-m
fi

launch_benchmark() {
    local name=$1
    local cmd=$2
    
    # Create new window
    tmux new-window -t $SESSION_NAME -n "$name"
    
    # Activate virtual environment
    tmux send-keys -t $SESSION_NAME:"$name" "source .venv/bin/activate" C-m
    
    # Run command
    echo "Launching $name: $cmd"
    tmux send-keys -t $SESSION_NAME:"$name" "$cmd" C-m
}

# Launch ConvLTC Benchmark
# We use smaller batch size because ConvLTC is memory intensive
launch_benchmark "conv_ltc" "python3 training/train_ltc.py --run_name conv_ltc_seq16_hybrid --model_type conv_ltc --epochs 50 --batch_size 4 --sequence_length 16 --lambda_physics 0.001"

echo "ConvLTC Benchmark launched in tmux session '$SESSION_NAME'"
echo "Attach with: tmux attach -t $SESSION_NAME"
