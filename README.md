# Real-Time Ultrasonic Temperature Monitoring for Tumor Ablation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)
![Platform](https://img.shields.io/badge/platform-Edge%20%7C%20Desktop-green)

This repository contains the implementation for the paper **"Hybrid CNN-LSTM and Physics-Informed Architectures for Real-Time Tumor Ablation Monitoring on Edge Devices"**.

## 🏥 Research Context

In non-invasive thermal ablation therapies (e.g., High-Intensity Focused Ultrasound - HIFU), precise temperature monitoring is critical to ensure tumor destruction while preserving healthy tissue. Direct temperature measurement is often impossible. This project implements a **non-invasive, privacy-preserving monitoring system** that estimates temperature dynamics directly from ultrasonic video sequences.

### Key Constraints & Goals
*   **Privacy-First:** Patient data is processed locally on edge devices.
*   **Cost-Effective:** Optimized for low-cost hardware (Raspberry Pi 4, Jetson Nano).
*   **Transparency:** Integrates **Uncertainty Quantification (UQ)**.

## 📚 Documentation
*   [**Methodology Part 1:** Single-Value Regression (Base)](Methodology_Part1.md)
*   [**Methodology Part 2:** Dense Temperature Map Estimation](Methodology_Part2.md)
*   [**Research Notes:** Advanced Architectures (LTCs) & Physics Priors](docs/RESEARCH_PART2.md)
*   [**Benchmark Status:** Current Results](BENCHMARK_STATUS.md)

## 🧠 Model Architectures

We evaluate and compare several architectures tailored for this task:

1.  **CNNLSTM (Hybrid):** A custom, lightweight architecture combining Convolutional Neural Networks for spatial feature extraction and LSTMs for temporal dynamics.
2.  **Physics-Informed CNN-LSTM:** Incorporates physical laws (temporal smoothness of heat diffusion) directly into the loss function.
3.  **ResNet-UNet (Dense):** Encoder-Decoder architecture for estimating full $64 \times 64$ temperature heatmaps from video frames.
4.  **Latent-LTC (ODES):** Advanced architecture using Liquid Time Constant networks to model the continuous-time physics of heat diffusion in a latent space.
5.  **Uncertainty Models:**
    *   **Deep Ensembles:** Multiple models trained with different initializations.
    *   **Bayesian Neural Networks:** Probabilistic weights to estimate epistemic uncertainty.

## ⚡ Edge Deployment & Clinical Demo

The project includes a full deployment pipeline using **ONNX Runtime** to ensure real-time performance on edge devices.

*   **Input:** 5-Channel Tensor (3 RGB + 2 Optical Flow)
*   **Performance:** >30 FPS on Raspberry Pi 4 / Jetson Nano
*   **Preprocessing:** Dense Optical Flow (Farneback) to capture heat shimmer and tissue changes.

## 🎥 Live Inference Demo

We provide a tool to visualize model predictions on video files in real-time. This is useful for creating demo videos or verifying model behavior.

### Features
*   **Real-Time Visualization:** Overlays predicted temperature, ground truth (if available), and error.
*   **FPS Counter:** Displays the current inference speed.
*   **ONNX Support:** Run inference using optimized ONNX models.
*   **Progress Bar:** Shows processing progress for long videos.

### Usage

**1. Run with PyTorch Model:**
```bash
# Run on a video file
python demo/live_inference.py \
    --video data/test_video.mp4 \
    --checkpoint models/cnnlstm_model.pth \
    --ground_truth data/test_temps.csv \
    --output demo_output.mp4

# Run on an image sequence (directory of images)
python demo/live_inference.py \
    --video data/sequence_1 \
    --checkpoint models/cnnlstm_model.pth \
    --output demo_output.mp4
```

**2. Run with ONNX Model (Faster):**
First, export your model:
```bash
python utils/export_to_onnx.py --model CNNLSTM --checkpoint models/cnnlstm_model.pth
```

Then run the demo:
```bash
python demo/live_inference.py \
    --video data/sequence_1 \
    --checkpoint models/onnx/CNNLSTM.onnx \
    --output demo_output_onnx.mp4
```

**3. Quick Test:**
To run a short test (first 150 frames / 5 seconds) to verify everything is working:
```bash
python demo/live_inference.py \
    --video data/sequence_1 \
    --checkpoint models/onnx/CNNLSTM.onnx \
    --quick_test
```

## 📂 Project Structure

```
.
├── models/                 # PyTorch model definitions (CNNLSTM, ResNet, Bayesian, etc.)
├── training/               # Training scripts and loops
├── simulation/             # Edge device simulation and profiling
│   ├── profiles.py         # Hardware specifications (RPi4, Jetson, etc.)
│   └── emulator.py         # Latency injection wrapper
├── utils/                  # Utility scripts
│   └── model_registry.py   # Centralized model configuration
├── benchmark_deployment.py # Main benchmarking script
├── convert_to_onnx.py      # ONNX export script
├── analyze_results.py      # Visualization of benchmark results
├── run_edge_benchmarks.sh  # Master script for full benchmark suite
├── paper/                  # LaTeX source for the research paper
├── data/                   # Dataset directory
├── logs/                   # Training logs

## 🚀 Edge Simulation & Benchmarking

We provide a comprehensive suite to simulate how these models perform on various edge hardware without needing the physical devices.

### Supported Simulated Devices
*   **Raspberry Pi 4 (4GB)** (CPU)
*   **NVIDIA Jetson Nano** (GPU)
*   **NVIDIA Jetson Orin Nano** (GPU)
*   **High-End Desktop GPU** (RTX 3090/4090)

### Running the Benchmarks

To run the full benchmark suite (PyTorch vs ONNX) and generate performance plots:

```bash
./run_edge_benchmarks.sh
```

This script will:
1.  Export all models to ONNX format.
2.  Run inference benchmarks on all simulated devices.
3.  Generate a CSV report (`results/edge_benchmark_results.csv`).
4.  Create visualization plots in `results/plots/`.

### Individual Scripts

*   **Convert Models:** `python convert_to_onnx.py`
*   **Run Benchmark:** `python benchmark_deployment.py --frames 50 --onnx`
*   **Analyze Results:** `python analyze_results.py`
├── run_benchmarks.sh       # Script to launch parallel training jobs
├── monitor_benchmarks.sh   # Dashboard to monitor training progress
└── requirements.txt        # Python dependencies
```

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/yourusername/video_regression.git
cd video_regression
pip install -r requirements.txt
```

### 2. Training Benchmarks

To reproduce the paper's results, run the full benchmark suite. This launches 7 parallel training jobs in a detached tmux session.

```bash
./run_benchmarks.sh
```

Monitor the progress using the dashboard:

```bash
./monitor_benchmarks.sh
```

### 3. Edge Performance Evaluation

To benchmark the models on your hardware (simulating edge constraints):

```bash
python demo/benchmark_edge.py
```

### 4. Clinical Demo

Run the simulated clinical monitoring pipeline:

```bash
python demo/run_clinical_demo.py
```

## 📊 Methodology

### Input Representation
We use a **5-channel input** to explicitly capture temporal dynamics:
*   **Channels 1-3:** RGB Video Frame (Spatial features)
*   **Channels 4-5:** Dense Optical Flow (dx, dy) (Temporal/Motion features)

### Physics-Informed Loss
$$ \mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda_{physics} \cdot \mathcal{L}_{physics} $$
We enforce consistency with **Newton's Law of Cooling**:
$$ \frac{dT}{dt} = -k(T - T_{env}) $$
This regularizes the model to learn physically plausible temperature trajectories, improving generalization in data-sparse regimes.

## 🔮 Further Development

Future work will focus on the following areas:

1.  **Advanced Bio-Heat Models:**
    *   Replace the simplified Newton's Law with the **Pennes' Bioheat Equation** to account for tissue perfusion and metabolic heat generation.
    *   Implement spatial regularization (diffusion) using Laplacian operators on the video frames.

2.  **Hardware Integration:**
    *   Deploy the optimized ONNX models on a physical **Raspberry Pi 4** with a connected USB ultrasound probe.
    *   Develop a lightweight GUI for real-time visualization on the edge device.

3.  **Clinical Validation:**
    *   Validate the model on **in-vivo** data from animal studies or clinical trials.
    *   Calibrate the uncertainty estimates (conformal prediction) to ensure valid coverage for clinical decision support.

4.  **Spatio-Temporal Attention:**
    *   Integrate attention mechanisms to focus the model on the specific region of interest (ROI) where the ablation is occurring.

## 📄 Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{neubuerger2025hybrid,
  title={Hybrid CNN-LSTM and Physics-Informed Architectures for Real-Time Tumor Ablation Monitoring on Edge Devices},
  author={Neubuerger, Felix and Nawrath, Helena},
  booktitle={Proceedings of the IEEE Conference on ...},
  year={2025}
}
```

## 📜 License

MIT License
