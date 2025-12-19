#!/bin/bash
# Launch Bayesian Spatial Convection PINN Benchmark
# This script runs in the background

LOG_FILE="logs/bayesian_spatial_convection.log"
mkdir -p logs

echo "Starting Bayesian Spatial Convection PINN Benchmark..." > $LOG_FILE
echo "Date: $(date)" >> $LOG_FILE

nohup /mnt/data2/video_regression/.venv/bin/python training/train_bayesian_spatial_convection.py --epochs 50 >> $LOG_FILE 2>&1 &

PID=$!
echo "Benchmark started with PID: $PID"
echo "Logs are being written to $LOG_FILE"
