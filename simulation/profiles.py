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
    # Slowdown factor relative to a High-End Desktop GPU (e.g., RTX 3090/4090)
    # Example: If Host takes 1ms, and factor is 50, Simulated Device takes 50ms.
    slowdown_factor: float 
    # Average Power Consumption in Watts (for energy estimation)
    power_watts: float
    # Variance or jitter in latency (std dev)
    latency_jitter: float = 0.005

# Define profiles
# Baselines estimated from ResNet50 inference times (see simulation/hardware_research.md):
# RTX 3090: ~1.8ms
# Jetson Orin Nano: ~11ms (Factor ~6.1x)
# Jetson Nano: ~36ms (Factor ~20x)
# RPi 4 (CPU): ~130ms (Factor ~72x)

PROFILES = {
    DeviceType.RASPBERRY_PI_4: DeviceProfile(
        name="Raspberry Pi 4 (4GB)",
        description="Quad-core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz. Running PyTorch CPU.",
        slowdown_factor=72.0, 
        power_watts=4.0, # Approx load power
        latency_jitter=0.020
    ),
    DeviceType.JETSON_NANO: DeviceProfile(
        name="NVIDIA Jetson Nano",
        description="128-core Maxwell GPU. Running PyTorch with CUDA (unoptimized).",
        slowdown_factor=20.0,
        power_watts=5.0, # 5W Mode
        latency_jitter=0.005
    ),
    DeviceType.JETSON_ORIN_NANO: DeviceProfile(
        name="NVIDIA Jetson Orin Nano",
        description="1024-core Ampere GPU. Running PyTorch with CUDA.",
        slowdown_factor=6.1,
        power_watts=7.0, # 7W Mode
        latency_jitter=0.002
    ),
    DeviceType.PC_CPU: DeviceProfile(
        name="Standard PC CPU",
        description="Intel/AMD Desktop CPU.",
        slowdown_factor=10.0, # CPU is significantly slower than GPU for CNNs
        power_watts=65.0,
        latency_jitter=0.005
    ),
    DeviceType.HIGH_END_GPU: DeviceProfile(
        name="High-End GPU (RTX 3090/4090)",
        description="Server grade GPU.",
        slowdown_factor=1.0, # Baseline
        power_watts=350.0,
        latency_jitter=0.0005
    )
}
