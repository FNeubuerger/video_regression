#!/bin/bash

# monitoring script for LOSO runs
# uses tmux capture-pane to get progress from all loso sessions

printf "\e[1;34m%-30s | %-15s | %-s\e[0m\n" "Session" "Fold" "Progress"
printf "%.s-" {1..100}
printf "\n"

# Only process sessions starting with 'loso_'
for session in $(tmux ls 2>/dev/null | grep "^loso_" | cut -d: -f1); do
    # Capture the full pane content
    output=$(tmux capture-pane -t "$session" -p)
    
    # Extract the current sequence folder (Fold)
    fold=$(echo "$output" | grep "FOLD: Holding out" | tail -n 1 | awk '{print $4}')
    [ -z "$fold" ] && fold="Initializing..."
    
    # Extract the progress bar/epoch info
    # Look for Epoch [x/y] line
    progress=$(echo "$output" | grep "Epoch \[" | tail -n 1 | xargs)
    [ -z "$progress" ] && progress="Starting up..."
    
    # Format and print
    printf "%-30s | %-15s | %-s\n" "$session" "$fold" "$progress"
done

printf "%.s-" {1..100}
printf "\n"
