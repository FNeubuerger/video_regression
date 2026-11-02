import pytest
import torch

from training.train import _align_regression_target


def test_align_regression_target_uses_last_sequence_label():
    outputs = torch.tensor([1.0, 2.0])
    labels = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])

    aligned = _align_regression_target(outputs, labels)

    torch.testing.assert_close(aligned, torch.tensor([12.0, 22.0]))


def test_align_regression_target_rejects_broadcasting_mismatch():
    with pytest.raises(ValueError, match="does not match target shape"):
        _align_regression_target(torch.zeros(2, 4), torch.zeros(2, 3))
