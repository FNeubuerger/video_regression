# Hardware Inference Speed Research

This document summarizes research into the inference capabilities of various edge devices compared to a high-end workstation GPU. These values are used to calibrate the `slowdown_factor` in our simulation environment.

## 1. Raspberry Pi 4 (4GB/8GB)
*   **Processor:** Broadcom BCM2711, Quad core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz.
*   **Inference Mode:** CPU (PyTorch/TFLite).
*   **Benchmark (ResNet-50):**
    *   ~130 ms per image (7.6 FPS) using TFLite XNNPACK.
    *   ~200+ ms per image using standard PyTorch.
*   **Benchmark (ResNet-18):**
    *   ~65 ms per image (15.4 FPS).
*   **Source:** [Q-engineering Deep Learning Benchmarks](https://qengineering.eu/deep-learning-examples-on-raspberry-32-64-os.html)

## 2. NVIDIA Jetson Nano
*   **Processor:** Quad-core ARM Cortex-A57 MPCore processor.
*   **GPU:** 128-core Maxwell.
*   **Inference Mode:** GPU (TensorRT FP16).
*   **Benchmark (ResNet-50):**
    *   ~36 ms per image (27 FPS).
*   **Benchmark (ResNet-18):**
    *   ~16 ms per image (62 FPS).
*   **Source:** [NVIDIA Jetson Nano Benchmarks](https://developer.nvidia.com/embedded/jetson-nano-dl-inference-benchmarks)

## 3. NVIDIA Jetson Orin Nano (8GB)
*   **Processor:** 6-core Arm® Cortex®-A78AE v8.2 64-bit CPU.
*   **GPU:** 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores.
*   **Inference Mode:** GPU (TensorRT INT8/FP16).
*   **Benchmark (ResNet-50):**
    *   ~11 ms per image (90 FPS) [INT8].
    *   ~25 ms per image (40 FPS) [FP16].
*   **Source:** [NVIDIA Jetson Orin Nano Compute Module](https://developer.nvidia.com/embedded/jetson-orin-nano-compute-module)

## 4. High-End GPU (NVIDIA RTX 3090)
*   **Architecture:** Ampere.
*   **Inference Mode:** GPU (PyTorch/TensorRT).
*   **Benchmark (ResNet-50):**
    *   ~1.8 ms per image (Batch Size 1).
*   **Source:** [Lambda Labs GPU Benchmarks](https://lambdalabs.com/gpu-benchmarks)

---

## Calculated Slowdown Factors

Based on the **ResNet-50** benchmarks (a common standard closest to our CNN-LSTM complexity), we derive the following slowdown factors relative to the RTX 3090 (1.8ms baseline).

| Device | Latency (ResNet-50) | Calculation | Estimated Factor |
| :--- | :--- | :--- | :--- |
| **RTX 3090** | 1.8 ms | 1.8 / 1.8 | **1.0x** |
| **Jetson Orin Nano** | 11 ms | 11 / 1.8 | **~6.1x** |
| **Jetson Nano** | 36 ms | 36 / 1.8 | **~20.0x** |
| **Raspberry Pi 4** | 130 ms | 130 / 1.8 | **~72.2x** |

*Note: These factors are approximations. Actual performance depends heavily on model architecture (e.g., LSTM layers might be even slower on CPU-bound devices like RPi4).*
