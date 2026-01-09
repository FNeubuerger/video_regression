#!/bin/bash

# run_final_evaluations.sh
# Runs all 16 model evaluations + Ensemble evaluation.
# Usage: ./run_final_evaluations.sh [parallel_jobs]
# Default parallel_jobs is 1 (sequential) to avoid OOM.

MAX_JOBS=${1:-1}
echo "Running evaluations with MAX_JOBS=$MAX_JOBS"

# Function to manage background jobs
function queue {
    while [[ $(jobs -r | wc -l) -ge $MAX_JOBS ]]; do
        sleep 1
    done
    "$@" &
}

# Ensure output directories exist
mkdir -p results/uncertainty_eval

echo "=== Starting Temporal Models (Deterministic) ==="

# 1. CNNLSTM
queue python evaluation/comprehensive_uncertainty_eval.py --model CNNLSTM --checkpoint models/cnnlstm_model.pth --samples 1

# 2. Pretrained CNNLSTM
queue python evaluation/comprehensive_uncertainty_eval.py --model PretrainedCNNLSTM --checkpoint models/pretrained_cnnlstm_model.pth --samples 1

# 3. Physics CNNLSTM
queue python evaluation/comprehensive_uncertainty_eval.py --model PhysicsCNNLSTM --checkpoint models/physics_cnnlstm_model.pth --samples 1

# 4. Bioheat PINN (Advanced Bioheat)
queue python evaluation/comprehensive_uncertainty_eval.py --model SpatialPhysicsCNNLSTM --checkpoint models/advanced_bioheat_model.pth --samples 1

# 5. Convection Bioheat
queue python evaluation/comprehensive_uncertainty_eval.py --model SpatialPhysicsCNNLSTM --checkpoint models/convection_bioheat_model.pth --samples 1

# 6. Metabolic Bioheat
queue python evaluation/comprehensive_uncertainty_eval.py --model SpatialPhysicsCNNLSTM --checkpoint models/metabolic_bioheat_model.pth --samples 1

echo "=== Starting Spatial Models (Deterministic) ==="

# 7. Simple ResNet
queue python evaluation/comprehensive_uncertainty_eval.py --model SimpleResNet --checkpoint models/simple_resnet_model.pth --samples 1

# 8. Spatial Bioheat
queue python evaluation/comprehensive_uncertainty_eval.py --model SpatialResNet --checkpoint models/spatial_bioheat_model.pth --samples 1

# 9. Spatial Convection
queue python evaluation/comprehensive_uncertainty_eval.py --model SpatialResNet --checkpoint models/spatial_convection_model.pth --samples 1

# 10. Spatial Metabolic
queue python evaluation/comprehensive_uncertainty_eval.py --model SpatialResNet --checkpoint models/spatial_metabolic_model.pth --samples 1

echo "=== Starting Uncertainty Models (Probabilistic) ==="

# 11. Bayesian Head (ResNet)
queue python evaluation/comprehensive_uncertainty_eval.py --model BayesianResNet --checkpoint checkpoints/bayesian_resnet_head.pth --samples 50

# 12. Full Bayesian ResNet
queue python evaluation/comprehensive_uncertainty_eval.py --model FullBayesianResNet --checkpoint checkpoints/full_bayesian_resnet.pth --samples 50

# 13. Bayesian PINN
queue python evaluation/comprehensive_uncertainty_eval.py --model BayesianResNet --checkpoint models/bayesian_pinn.pth --samples 50

# 14. Bayesian CNNLSTM
queue python evaluation/comprehensive_uncertainty_eval.py --model BayesianCNNLSTM --checkpoint models/bayesian_cnnlstm.pth --samples 50

# 15. Bayesian Metabolic PINN
queue python evaluation/comprehensive_uncertainty_eval.py --model BayesianCNNLSTM --checkpoint models/bayesian_metabolic_pinn.pth --samples 50

# 16. Bayesian Spatial Convection
queue python evaluation/comprehensive_uncertainty_eval.py --model BayesianSpatialResNet --checkpoint models/bayesian_spatial_convection.pth --samples 50

echo "=== Starting Ensemble Evaluation ==="

# 17. Ensemble (SimpleResNet)
# Assumes checkpoints are in checkpoints/ensemble/
queue python evaluation/comprehensive_uncertainty_eval.py --model SimpleResNet --ensemble_dir checkpoints/ensemble --samples 1

# Wait for all jobs to finish
wait

echo "All evaluations complete. Results saved in results/uncertainty_eval/"
