# Clinical Inference Demo

This directory contains scripts to demonstrate the deployment of the trained video regression models in a simulated clinical environment.

## Contents

- `export_models.py`: Exports trained PyTorch models to ONNX format for efficient inference.
- `run_clinical_demo.py`: Simulates a clinical session by processing a video sequence frame-by-frame, performing preprocessing (including optical flow), and running inference using ONNX Runtime.

## Usage

### 1. Export Models to ONNX

First, export the trained models. If checkpoints are not available, it will export untrained models with random weights for testing purposes.

```bash
python demo/export_models.py
```

This will create `.onnx` files in the `demo/` directory.

### 2. Run the Clinical Demo

Run the demo with a specific model:

```bash
python demo/run_clinical_demo.py --model demo/cnnlstm.onnx
```

You can also try other models:
- `demo/pretrained_cnnlstm.onnx`
- `demo/simple_resnet.onnx`
- `demo/physics_cnnlstm.onnx`

## Performance

The demo measures the end-to-end latency per frame, including:
1.  **Preprocessing**: Resizing, Normalization, Optical Flow calculation (Farneback).
2.  **Inference**: ONNX Runtime execution (CPU).

The script outputs the latency (ms) and FPS for each frame, and the average at the end.

## Simulating Limited Resources

The demo uses `CPUExecutionProvider` by default, which gives a good baseline for performance on devices without dedicated GPUs (like Raspberry Pi, though a Pi is slower than a server CPU).
