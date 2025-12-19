from models.backbones import CNNLSTM, PretrainedCNNLSTM
from physics.models import PhysicsCNNLSTM, SpatialPhysicsCNNLSTM
from models.bayesian import BayesianResNet, BayesianCNNLSTM, FullBayesianResNet, BayesianSpatialResNet
from torchvision.models import resnet18

# Maps model name to (Class, Args)
MODEL_REGISTRY = {
    "CNNLSTM": (CNNLSTM, {"frame_shape": (64, 64, 5), "time_steps": 5}),
    "PretrainedCNNLSTM": (PretrainedCNNLSTM, {"pretrained_cnn": resnet18(weights='IMAGENET1K_V1'), "frame_shape": (64, 64, 5), "time_steps": 5}),
    "PhysicsCNNLSTM": (PhysicsCNNLSTM, {"frame_shape": (64, 64, 5), "time_steps": 5}),
    "SpatialPhysicsCNNLSTM": (SpatialPhysicsCNNLSTM, {"frame_shape": (64, 64, 5), "time_steps": 5}),
    "BayesianResNet": (BayesianResNet, {"frame_shape": (64, 64, 5)}),
    "BayesianCNNLSTM": (BayesianCNNLSTM, {"frame_shape": (64, 64, 5)}),
    "FullBayesianResNet": (FullBayesianResNet, {"frame_shape": (64, 64, 5)}),
    "BayesianSpatialResNet": (BayesianSpatialResNet, {"frame_shape": (64, 64, 5)})
}
