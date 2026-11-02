# Demos

All demos take a model name from `utils/model_registry.py` and a checkpoint
(`.pth`, or `.onnx` where noted). Checkpoints are not tracked in git.

## Export to ONNX

```bash
python utils/export_to_onnx.py --model CNNLSTM --checkpoint models/cnnlstm_model.pth
# -> models/cnnlstm_model.onnx (override with --output)
```

## Live inference (`live_inference.py`)

Overlays predicted temperature, optional ground truth, error and FPS on a video
or an image-sequence directory. Accepts `.pth` or `.onnx` checkpoints.

```bash
python demo/live_inference.py --video data/sequence_1 \
    --checkpoint models/cnnlstm_model.onnx --output demo_output.mp4 [--quick_test]
```

## Clinical session simulation (`run_clinical_demo.py`)

Frame-by-frame preprocessing (resize, normalization, Farneback flow) plus
inference; reports per-frame latency and FPS.

```bash
python demo/run_clinical_demo.py --model CNNLSTM \
    --checkpoint models/cnnlstm_model.pth --sequence data/sequence_1
```

## XAI dashboard (`xai_dashboard.py`)

Side-by-side prediction and attribution video for dense models (default `ResNetUNet`).
Requires `captum`.

```bash
python demo/xai_dashboard.py --video data/sequence_1 --model ResNetUNet \
    --checkpoint checkpoints/unet_sparse/best_model.pth
```

## Other

- `visualize_results.py` — dashboard video for dense/LTC models
  (default `LatentLTC_UNet`, `checkpoints/ltc_unet/best_model.pth`).
- `visualize_roi_context.py` — ROI context visualization.
