import numpy as np
import pytest
import torch
from torch import nn

from evaluation import evaluate_uncertainty


class _FakeResNet(nn.Module):
    def __init__(self, frame_shape, output_dim):
        super().__init__()
        channels = frame_shape[2]
        self.backbone = nn.Module()
        self.backbone.conv1 = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.regressor = nn.Linear(1, output_dim)

    def forward(self, images):
        features = self.backbone.conv1(images).mean(dim=(2, 3))
        return self.regressor(features).squeeze(-1)


@pytest.mark.parametrize("output_dim", [1, 4])
def test_ensemble_infers_checkpoint_dimensions_and_target_shape(
    tmp_path, monkeypatch, output_dim
):
    monkeypatch.setattr(evaluate_uncertainty, "SimpleResNet", _FakeResNet)
    for model_idx in range(2):
        model = _FakeResNet((4, 4, 5), output_dim)
        torch.save(model.state_dict(), tmp_path / f"model_{model_idx}.pth")

    images = torch.ones(2, 3, 5, 4, 4)
    scalars = torch.tensor(
        [
            [[1.0, 3.0, 5.0, 7.0]] * 3,
            [[2.0, 4.0, 6.0, 8.0]] * 3,
        ]
    )
    batch = (images, torch.zeros(2, 3, 1, 4, 4), torch.zeros(2, 3, 1, 4, 4), scalars)

    means, stds, targets = evaluate_uncertainty.evaluate_ensemble(
        [batch], num_models=2, device="cpu", checkpoint_dir=str(tmp_path)
    )

    expected_shape = (2,) if output_dim == 1 else (2, 4)
    assert means.shape == targets.shape == expected_shape
    assert stds.shape == means.shape
    expected_targets = (
        scalars[:, -1].mean(dim=1).numpy()
        if output_dim == 1
        else scalars[:, -1].numpy()
    )
    np.testing.assert_allclose(targets, expected_targets)


def test_ensemble_rejects_mismatched_target_dimensions():
    with pytest.raises(ValueError, match="Unsupported ensemble output dimension"):
        evaluate_uncertainty._ensemble_targets(torch.zeros(2, 4), output_dim=2)
