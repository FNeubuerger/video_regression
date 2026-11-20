import pytest
import torch

from evaluation.loso_metrics import batch_errors
from training.loso_probe_training import measured_probe_errors, split_by_sequence


def test_sequence_predictions_pair_by_window_and_ignore_missing_probes():
    images = torch.randn(2, 5, 5, 4, 4)
    target = torch.full((2, 5, 4), 500.0)
    target[:, -1] = torch.tensor([[10, 20, float("nan"), 30], [40, 50, 60, 70]])
    pred = torch.zeros_like(target)
    pred[:, -1] = torch.tensor([[11, 22, 1000, 33], [36, 46, 56, 66]])
    batch = (images, None, torch.full((2, 5, 1, 4, 4), 37.0), target, None)

    def model(inputs):
        torch.testing.assert_close(inputs, images)
        return pred, torch.tensor(0.0)

    scalar, partial, probe = batch_errors(model, batch, "cpu", spatial=False)
    torch.testing.assert_close(scalar, torch.tensor([-4.0]))
    torch.testing.assert_close(partial, torch.tensor([2.0]))
    torch.testing.assert_close(probe, torch.tensor([1, 2, 3, -4, -4, -4, -4.0]))
    with pytest.raises(ValueError, match="Unexpected temperature output shape"):
        batch_errors(lambda _: pred.mean(-1), batch, "cpu", spatial=False)


def test_spatial_scoring_includes_measured_zero_but_not_unmeasured_pixels():
    images = torch.zeros(1, 5, 5, 4, 4)
    target = torch.full((1, 5, 4), float("nan"))
    target[0, -1, :2] = torch.tensor([0.0, 4.0])
    heatmaps = torch.zeros(1, 5, 1, 4, 4)
    heatmaps[0, -1, 0, 0, 1] = 4
    valid = torch.zeros_like(heatmaps, dtype=torch.bool)
    valid[0, -1, 0, 0, :2] = True
    fields = torch.full((1, 5, 1, 2, 2), 3.0)
    fields[:, :-1] = 99
    scalar, partial, probe = batch_errors(
        lambda _: fields, (images, heatmaps, None, target, valid), "cpu", spatial=True
    )
    assert scalar.numel() == 0
    torch.testing.assert_close(partial, torch.tensor([1.0]))
    torch.testing.assert_close(probe, torch.tensor([3.0, -1.0]))


def test_training_pairs_final_frame_probes_and_masks_missing_gradients():
    pred = torch.tensor([[2.0, 40.0, 5.0, 80.0], [1.0, 2.0, 3.0, 4.0]])
    pred.requires_grad_()
    targets = torch.tensor(
        [
            [[100.0] * 4, [0.0, float("nan"), 6.0, float("inf")]],
            [[100.0] * 4, [2.0, 2.0, 3.0, 0.0]],
        ]
    )
    errors = measured_probe_errors(pred, targets)
    torch.testing.assert_close(errors, torch.tensor([2.0, -1.0, -1.0, 0.0, 0.0, 4.0]))
    errors.square().sum().backward()
    torch.testing.assert_close(
        pred.grad, torch.tensor([[4.0, 0.0, -2.0, 0.0], [-2.0, 0.0, 0.0, 8.0]])
    )
    with pytest.raises(ValueError, match="probe outputs"):
        measured_probe_errors(pred.flatten(), targets)


def test_validation_holds_out_whole_sequences_reproducibly():
    windows = [(f"seq_{i % 5}", i) for i in range(23)]
    train, val, val_sequences = split_by_sequence(windows, seed=42)
    assert (train, val, val_sequences) == split_by_sequence(windows, seed=42)
    assert sorted(train + val) == list(range(23))
    assert {windows[i][0] for i in val} == set(val_sequences)
    assert {windows[i][0] for i in train}.isdisjoint(val_sequences)
    with pytest.raises(ValueError, match="at least two"):
        split_by_sequence([("seq_a", 1), ("seq_a", 2)], seed=42)
