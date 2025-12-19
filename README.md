# Real-Time Ultrasonic Temperature Monitoring for Tumor Ablation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)
![Platform](https://img.shields.io/badge/platform-Edge%20%7C%20Desktop-green)

This repository contains the implementation for the paper **"Hybrid CNN-LSTM and Physics-Informed Architectures for Real-Time Tumor Ablation Monitoring on Edge Devices"**.

## 🏥 Research Context

In non-invasive thermal ablation therapies (e.g., High-Intensity Focused Ultrasound - HIFU), precise temperature monitoring is critical to ensure tumor destruction while preserving healthy tissue. Direct temperature measurement is often impossible. This project implements a **non-invasive, privacy-preserving monitoring system** that estimates temperature dynamics directly from ultrasonic video sequences.

### Key Constraints & Goals
*   **Privacy-First:** Patient data is processed locally on edge devices, eliminating the need for cloud transmission.
*   **Cost-Effective:** Optimized for low-cost hardware (Raspberry Pi 4, Jetson Nano) to allow easy retrofitting in clinical settings.
*   **Transparency:** Integrates **Uncertainty Quantification (UQ)** to provide clinicians with confidence intervals alongside predictions.

## 🧠 Model Architectures

We evaluate and compare several architectures tailored for this task:

1.  **CNNLSTM (Hybrid):** A custom, lightweight architecture combining Convolutional Neural Networks for spatial feature extraction and LSTMs for temporal dynamics.
2.  **Physics-Informed CNN-LSTM:** Incorporates physical laws (temporal smoothness of heat diffusion) directly into the loss function.
3.  **Pretrained ResNet:** A transfer learning approach using ImageNet features.
4.  **Uncertainty Models:**
    *   **Deep Ensembles:** Multiple models trained with different initializations.
    *   **Bayesian Neural Networks:** Probabilistic weights to estimate epistemic uncertainty.

## ⚡ Edge Deployment & Clinical Demo

The project includes a full deployment pipeline using **ONNX Runtime** to ensure real-time performance on edge devices.

*   **Input:** 5-Channel Tensor (3 RGB + 2 Optical Flow)
*   **Performance:** >30 FPS on Raspberry Pi 4 / Jetson Nano
*   **Preprocessing:** Dense Optical Flow (Farneback) to capture heat shimmer and tissue changes.

## 📂 Project Structure

```
.
├── models/                 # PyTorch model definitions (CNNLSTM, ResNet, Bayesian, etc.)
├── training/               # Training scripts and loops
├── demo/                   # Edge deployment and clinical simulation
│   ├── benchmark_edge.py   # Latency/FPS benchmarking script
│   ├── export_models.py    # ONNX export utilities
│   └── run_clinical_demo.py # Real-time inference simulation
├── paper/                  # LaTeX source for the research paper
├── data/                   # Dataset directory
├── logs/                   # Training logs
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
