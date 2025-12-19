from dataclasses import dataclass
from enum import Enum

class DeviceType(Enum):
    RASPBERRY_PI_4 = "rpi4"
    JETSON_NANO = "jetson_nano"
    JETSON_ORIN_NANO = "jetson_orin_nano"
    PC_CPU = "pc_cpu"
    HIGH_END_GPU = "high_end_gpu"

@dataclass
class DeviceProfile:
    name: str
    description: str
    # Expected latency in seconds for a ResNet18-like backbone (single frame)
    # These are approximate values for simulation purposes
    base_latency_resnet18: float 
    # Variance or jitter in latency (std dev)
    latency_jitter: float = 0.005

# Define profiles
PROFILES = {
    DeviceType.RASPBERRY_PI_4: DeviceProfile(
        name="Raspberry Pi 4 (4GB)",
        description="Quad-core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz. Running PyTorch CPU.",
        base_latency_resnet18=0.150, # ~150ms (approx 6-7 FPS)
        latency_jitter=0.020
    ),
    DeviceType.JETSON_NANO: DeviceProfile(
        name="NVIDIA Jetson Nano",
        description="128-core Maxwell GPU. Running PyTorch with CUDA (unoptimized).",
        base_latency_resnet18=0.045, # ~45ms (approx 22 FPS)
        latency_jitter=0.005
    ),
    DeviceType.JETSON_ORIN_NANO: DeviceProfile(
        name="NVIDIA Jetson Orin Nano",
        description="1024-core Ampere GPU. Running PyTorch with CUDA.",
        base_latency_resnet18=0.010, # ~10ms (approx 100 FPS)
        latency_jitter=0.002
    ),
    DeviceType.PC_CPU: DeviceProfile(
        name="Standard PC CPU",
        description="Intel/AMD Desktop CPU.",
        base_latency_resnet18=0.030, # ~30ms
        latency_jitter=0.005
    ),
    DeviceType.HIGH_END_GPU: DeviceProfile(
        name="High-End GPU (RTX 3090/4090)",
        description="Server grade GPU.",
        base_latency_resnet18=0.003, # ~3ms
        latency_jitter=0.0005
    )
}
