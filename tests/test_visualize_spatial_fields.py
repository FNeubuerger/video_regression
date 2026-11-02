import matplotlib
import pytest
import torch

matplotlib.use("Agg")

from evaluation import visualize_spatial_fields
from evaluation.visualize_spatial_fields import _last_field_batch


def test_last_field_batch_selects_final_frame_and_removes_single_channel():
    fields = torch.arange(2 * 3 * 1 * 4 * 5).reshape(2, 3, 1, 4, 5)

    result = _last_field_batch(fields, sequence_length=3)

    assert result.shape == (2, 4, 5)
    torch.testing.assert_close(result, fields[:, -1, 0])


def test_last_field_batch_accepts_temporal_fields_without_channel_axis():
    fields = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

    result = _last_field_batch(fields, sequence_length=3)

    assert result.shape == (2, 4, 5)
    torch.testing.assert_close(result, fields[:, -1])


def test_last_field_batch_rejects_unexpected_dimensions():
    with pytest.raises(ValueError, match="Expected a batch of 2D fields"):
        _last_field_batch(torch.zeros(2, 3, 4, 5, 6, 7), sequence_length=3)


def test_spatial_visualization_includes_prediction_and_ground_truth_panels(
    monkeypatch,
):
    captured_axes = {}
    original_subplots = visualize_spatial_fields.plt.subplots

    def fake_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        captured_axes["axes"] = axes
        return figure, axes

    class FakeModel(torch.nn.Module):
        def forward(self, frames):
            return torch.ones(frames.shape[0], frames.shape[1], 4, 4)

    monkeypatch.setattr(visualize_spatial_fields.plt, "subplots", fake_subplots)
    monkeypatch.setattr(visualize_spatial_fields.plt, "savefig", lambda *a, **k: None)
    monkeypatch.setattr(
        visualize_spatial_fields,
        "load_spatial_model",
        lambda *a: (FakeModel(), "cpu"),
    )
    monkeypatch.setattr(
        visualize_spatial_fields,
        "get_sample_batch",
        lambda batch_size: (
            torch.zeros(batch_size, 5, 5, 8, 8),
            torch.ones(batch_size, 5, 1, 8, 8),
        ),
    )

    visualize_spatial_fields.visualize_spatial_predictions()

    titles = [ax.get_title() for ax in captured_axes["axes"].flat]
    assert sum("Prediction" in title for title in titles) == 6
    assert sum("Sensor ground truth" in title for title in titles) == 6
