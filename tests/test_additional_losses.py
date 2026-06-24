"""Smoke tests for the new additional physics losses."""

import torch

from physics.additional_losses import (
    AdvectionDiffusionLoss,
    ArrheniusDamageLoss,
    CompositePhysicsLoss,
    EnergyConservationLoss,
    MaximumPrincipleLoss,
    ParameterTVLoss,
    RadialSymmetryLoss,
    SensorPinningLoss,
)


def _seq(batch=2, time=4, h=8, w=8):
    return torch.linspace(20, 70, batch * time * h * w).reshape(batch, time, h, w)


def test_energy_loss_runs_and_is_scalar():
    out = EnergyConservationLoss()(torch.randn(2, 4, 8, 8))
    assert out.dim() == 0


def test_max_principle_zero_for_constant_field():
    T = torch.ones(2, 3, 5, 5) * 42
    assert MaximumPrincipleLoss()(T).item() == 0.0


def test_sensor_pinning():
    T = torch.zeros(1, 2, 8, 8)
    coords = [(1, 1), (2, 2)]
    vals = torch.tensor([[[3.0, 5.0], [3.0, 5.0]]])  # (B, T, N)
    loss = SensorPinningLoss(weight=1.0)(T, coords, vals)
    assert loss.item() > 0


def test_advection_runs():
    T = _seq()
    flow = torch.zeros(2, 4, 2, 8, 8)
    out = AdvectionDiffusionLoss()(T, flow)
    assert out.dim() == 0


def test_radial_symmetry_zero_for_constant():
    T = torch.ones(1, 2, 16, 16) * 30
    assert RadialSymmetryLoss()(T).item() == 0.0


def test_arrhenius_runs():
    T = _seq()
    out = ArrheniusDamageLoss()(T)
    assert out.dim() == 0


def test_tv_loss_positive():
    p = torch.randn(8, 8)
    assert ParameterTVLoss(weight=1.0)(p).item() > 0


def test_composite_runs():
    T = _seq()
    flow = torch.zeros(2, 4, 2, 8, 8)
    loss = CompositePhysicsLoss(
        w_energy=1.0,
        w_max_principle=1.0,
        w_advection=1.0,
        w_radial=0.1,
        w_arrhenius=1.0,
    )(T, flow=flow, antenna_centre=(4, 4))
    assert loss.dim() == 0
