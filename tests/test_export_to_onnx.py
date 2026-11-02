"""ONNX export: checkpoint formats and PyTorch/onnxruntime equivalence."""

import numpy as np
import onnxruntime as ort
import pytest
import torch

from utils.export_to_onnx import export_to_onnx
from utils.model_registry import MODEL_REGISTRY


def _build(name):
    torch.manual_seed(0)
    model_class, kwargs = MODEL_REGISTRY[name]
    return model_class(**kwargs).eval()


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("name", ["SimpleResNet", "ResNetUNet"])
def test_export_matches_pytorch(tmp_path, name, wrapped):
    model = _build(name)
    state = model.state_dict()
    ckpt = tmp_path / "model.pth"
    torch.save({"epoch": 1, "model_state_dict": state} if wrapped else state, ckpt)
    onnx_path = tmp_path / "model.onnx"

    export_to_onnx(name, str(ckpt), str(onnx_path))

    session = ort.InferenceSession(str(onnx_path))
    shape = [d if isinstance(d, int) else 2 for d in session.get_inputs()[0].shape]
    x = torch.randn(*shape)
    with torch.no_grad():
        expected = model(x)
    if isinstance(expected, (tuple, list)):
        expected = expected[0]
    (actual,) = session.run(None, {"input": x.numpy()})[:1]
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-3, atol=1e-4)
