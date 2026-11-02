#!/bin/bash
# Restart Bayesian Physics Benchmark Stream

# 1. Bayesian PINN
echo "Starting Bayesian PINN Training..."
nohup python3 training/train_bayesian_pinn.py --epochs 30 --batch_size 16 > logs/bayesian_pinn_restart.log 2>&1 &
PID1=$!
echo "Bayesian PINN started with PID $PID1"

# 2. Bayesian Convection PINN (wait a bit or run in parallel if memory allows)
# We use smaller batch size to be safe
echo "Starting Bayesian Convection PINN Training..."
nohup python3 training/train_bayesian_convection_pinn.py --epochs 30 --batch_size 16 > logs/bayesian_convection_restart.log 2>&1 &
PID2=$!
echo "Bayesian Convection PINN started with PID $PID2"

echo "Bayesian Benchmarks Restarted. Logs in logs/bayesian_*_restart.log"
