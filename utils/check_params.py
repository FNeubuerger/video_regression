import torch
from physics.models import SpatialPhysicsCNNLSTM

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# frame_shape=(H, W, C)
model = SpatialPhysicsCNNLSTM(frame_shape=(64, 64, 2), time_steps=10)
params = count_parameters(model)
print(f"SpatialPhysicsCNNLSTM Parameters: {params}")
print(f"SpatialPhysicsCNNLSTM Parameters (M): {params/1e6:.2f}")
