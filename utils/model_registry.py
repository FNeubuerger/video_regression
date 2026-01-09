from models.backbones import CNNLSTM, PretrainedCNNLSTM, SpatialResNet, SimpleResNet
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
    "BayesianSpatialResNet": (BayesianSpatialResNet, {"frame_shape": (64, 64, 5)}),
    "SimpleResNet": (SimpleResNet, {"frame_shape": (64, 64, 5)}),
    "SpatialResNet": (SpatialResNet, {"frame_shape": (64, 64, 5)})
}

# --- Part 2 & 3 Models ---
try:
    from models.dense_heads import ResNetUNet
    from models.latent_ltc import LatentLTC_UNet
    from models.conv_ltc import ConvLTC_Model

    # NOTE: ResNetUNet takes n_channels=3 (RGB)
    MODEL_REGISTRY["ResNetUNet"] = (ResNetUNet, {"n_channels": 3, "n_classes": 1, "variational": False})
    
    # NOTE: LatentLTC_UNet takes Sequence inputs
    # The trained checkpoint from Part 3 used n_channels=3, 160 units, and VAE
    MODEL_REGISTRY["LatentLTC_UNet"] = (LatentLTC_UNet, {"n_channels": 3, "latent_dim": 128, "ncp_units": 160, "variational": True})

    # ConvLTC
    MODEL_REGISTRY["ConvLTC"] = (ConvLTC_Model, {"input_channels": 3, "hidden_channels": 32, "output_channels": 1})

except ImportError as e:
    print(f"Warning: Could not import Part 2/3 models: {e}")

