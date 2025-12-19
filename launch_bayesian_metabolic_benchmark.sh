#!/bin/bash
# Launch Bayesian Metabolic PINN Benchmark
# This script runs in the background

LOG_FILE="logs/bayesian_metabolic_pinn.log"
mkdir -p logs

echo "Starting Bayesian Metabolic PINN Benchmark..." > $LOG_FILE
echo "Date: $(date)" >> $LOG_FILE

nohup python training/train_bayesian_metabolic_pinn.py --epochs 50 >> $LOG_FILE 2>&1 &

PID=$!
echo "Benchmark started with PID: $PID"
echo "Logs are being written to $LOG_FILE"
