#!/bin/bash

SESSION="video_regression_eval"
WORKSPACE="/mnt/data2/video_regression"

# Create session if it doesn't exist
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists. Attaching..."
else
    echo "Creating new session: $SESSION"
    tmux new-session -d -s "$SESSION" -n "monitor"
fi

# Function to create a window and run commands
launch_eval_group() {
    WINDOW_NAME=$1
    COMMANDS=$2
    
    # Check if window exists
    if tmux list-windows -t $SESSION | grep -q "$WINDOW_NAME"; then
        echo "Window $WINDOW_NAME already exists. Skipping..."
    else
        tmux new-window -t $SESSION -n "$WINDOW_NAME"
        tmux send-keys -t $SESSION:$WINDOW_NAME "cd $WORKSPACE" C-m
        tmux send-keys -t $SESSION:$WINDOW_NAME "source .venv/bin/activate" C-m
        tmux send-keys -t $SESSION:$WINDOW_NAME "$COMMANDS" C-m
    fi
}

# 1. Temporal Models (Deterministic - Fast)
CMD_TEMPORAL="
python evaluation/comprehensive_uncertainty_eval.py --model CNNLSTM --checkpoint models/cnnlstm_model.pth --samples 1;
python evaluation/comprehensive_uncertainty_eval.py --model PretrainedCNNLSTM --checkpoint models/pretrained_cnnlstm_model.pth --samples 1;
python evaluation/comprehensive_uncertainty_eval.py --model PhysicsCNNLSTM --checkpoint models/physics_cnnlstm_model.pth --samples 1;
python evaluation/comprehensive_uncertainty_eval.py --model SpatialPhysicsCNNLSTM --checkpoint models/advanced_bioheat_model.pth --samples 1;
python evaluation/comprehensive_uncertainty_eval.py --model SpatialPhysicsCNNLSTM --checkpoint models/convection_bioheat_model.pth --samples 1;
python evaluation/comprehensive_uncertainty_eval.py --model SpatialPhysicsCNNLSTM --checkpoint models/metabolic_bioheat_model.pth --samples 1;
echo 'Temporal evaluations complete.'
"
launch_eval_group "eval_temporal" "$CMD_TEMPORAL"

# 2. Spatial Models (Deterministic - Fast)
CMD_SPATIAL="
python evaluation/comprehensive_uncertainty_eval.py --model SimpleResNet --checkpoint models/simple_resnet_model.pth --samples 1;
python evaluation/comprehensive_uncertainty_eval.py --model SpatialResNet --checkpoint models/spatial_bioheat_model.pth --samples 1;
python evaluation/comprehensive_uncertainty_eval.py --model SpatialResNet --checkpoint models/spatial_convection_model.pth --samples 1;
python evaluation/comprehensive_uncertainty_eval.py --model SpatialResNet --checkpoint models/spatial_metabolic_model.pth --samples 1;
echo 'Spatial evaluations complete.'
"
launch_eval_group "eval_spatial" "$CMD_SPATIAL"

# 3. Uncertainty Models Group A (Probabilistic - Slow)
# Note: Corrected checkpoint path for BayesianResNet to checkpoints/bayesian_resnet.pth
CMD_UNCERTAINTY_A="
python evaluation/comprehensive_uncertainty_eval.py --model BayesianResNet --checkpoint checkpoints/bayesian_resnet.pth --samples 50;
python evaluation/comprehensive_uncertainty_eval.py --model FullBayesianResNet --checkpoint checkpoints/full_bayesian_resnet.pth --samples 50;
python evaluation/comprehensive_uncertainty_eval.py --model BayesianResNet --checkpoint models/bayesian_pinn.pth --samples 50;
echo 'Uncertainty Group A complete.'
"
launch_eval_group "eval_uq_a" "$CMD_UNCERTAINTY_A"

# 4. Uncertainty Models Group B (Probabilistic - Slow)
CMD_UNCERTAINTY_B="
python evaluation/comprehensive_uncertainty_eval.py --model BayesianCNNLSTM --checkpoint models/bayesian_cnnlstm.pth --samples 50;
python evaluation/comprehensive_uncertainty_eval.py --model BayesianCNNLSTM --checkpoint models/bayesian_metabolic_pinn.pth --samples 50;
python evaluation/comprehensive_uncertainty_eval.py --model BayesianSpatialResNet --checkpoint models/bayesian_spatial_convection.pth --samples 50;
echo 'Uncertainty Group B complete.'
"
launch_eval_group "eval_uq_b" "$CMD_UNCERTAINTY_B"

# 5. Ensemble (Moderate)
CMD_ENSEMBLE="
python evaluation/comprehensive_uncertainty_eval.py --model SimpleResNet --ensemble_dir checkpoints/ensemble --samples 1;
echo 'Ensemble evaluation complete.'
"
launch_eval_group "eval_ensemble" "$CMD_ENSEMBLE"

echo "All evaluation groups launched in tmux session '$SESSION'."
echo "Attach with: tmux attach -t $SESSION"
