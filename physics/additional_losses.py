"""Additional physics-informed losses for ablation-zone temperature modelling.

The legacy code base already implements:

* ``physics.loss.PhysicsInformedLoss``: Newton's cooling residual on scalar
  temperature sequences.
* ``physics.bioheat_loss.AdvancedBioHeatLoss``: Pennes bioheat residual on
  spatial fields, optionally with learnable perfusion / conductivity.

This module adds the losses that were missing and that came up during the
academic audit (see ACADEMIC_AUDIT.md).  Every loss is a small
``nn.Module`` so they compose with the existing ones via simple addition:

    >>> total = mse + 0.1 * energy_loss(T) + 0.05 * arrhenius_loss(T)

All losses are written to be safe on both
``(B, T, H, W)`` spatial sequences and ``(B, T)`` scalar sequences;
they no-op when applied to a tensor with the wrong rank, returning
``torch.zeros((), device=T.device)``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_spatial_seq(T: torch.Tensor) -> bool:
    return T.dim() == 4  # (B, T, H, W)


def _zero(T: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=T.device, dtype=T.dtype)


def _laplacian(T: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    """5-point Laplacian on (B, T, H, W) with replicate padding."""

    T_pad = F.pad(T, (1, 1, 1, 1), mode="replicate")
    lap = (
        T_pad[:, :, 1:-1, :-2]
        + T_pad[:, :, 1:-1, 2:]
        + T_pad[:, :, :-2, 1:-1]
        + T_pad[:, :, 2:, 1:-1]
        - 4.0 * T
    ) / (dx**2)
    return lap


def _grad_xy(T: torch.Tensor, dx: float = 1.0):
    T_pad = F.pad(T, (1, 1, 1, 1), mode="replicate")
    T_x = (T_pad[:, :, 1:-1, 2:] - T_pad[:, :, 1:-1, :-2]) / (2 * dx)
    T_y = (T_pad[:, :, 2:, 1:-1] - T_pad[:, :, :-2, 1:-1]) / (2 * dx)
    return T_x, T_y


# ---------------------------------------------------------------------------
# 1. Energy-conservation loss
# ---------------------------------------------------------------------------


class EnergyConservationLoss(nn.Module):
    r"""Penalise violation of the integral energy balance.

    For an insulated ROI with volumetric source :math:`Q`:

    .. math::
        \frac{d}{dt}\int_\Omega T\,dV
        = \int_\Omega Q\,dV
        + \oint_{\partial\Omega} k\nabla T \cdot \hat n\,dA.

    With zero-flux boundary the surface integral vanishes, so the mean
    temperature must increase at a rate proportional to the spatial mean
    source.  When ``Q`` is unknown we assume it is approximately constant
    across the heating phase and penalise non-monotonic accumulation of
    thermal energy.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self, T: torch.Tensor, source: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not _is_spatial_seq(T) or T.shape[1] < 2:
            return _zero(T)
        # Mean temperature per frame
        mean_T = T.mean(dim=(2, 3))  # (B, T)
        dE = mean_T[:, 1:] - mean_T[:, :-1]
        if source is None:
            # During heating dE/dt should be >= 0 (monotone increasing energy)
            residual = F.relu(-dE)
        else:
            mean_Q = source.mean(dim=(2, 3))[:, :-1]
            residual = dE - mean_Q
        return self.weight * residual.pow(2).mean()


# ---------------------------------------------------------------------------
# 2. Maximum-principle loss
# ---------------------------------------------------------------------------


class MaximumPrincipleLoss(nn.Module):
    """For source-free interior, T cannot exceed neighbouring extrema.

    We penalise pixels whose value exceeds the local 3x3 max of the
    previous frame (forward heat equation maximum principle).
    """

    def __init__(self, weight: float = 1.0, kernel: int = 3):
        super().__init__()
        self.weight = weight
        self.kernel = kernel

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        if not _is_spatial_seq(T) or T.shape[1] < 2:
            return _zero(T)
        B, Tn, H, W = T.shape
        prev = T[:, :-1].reshape(B * (Tn - 1), 1, H, W)
        local_max = F.max_pool2d(prev, self.kernel, stride=1, padding=self.kernel // 2)
        local_max = local_max.view(B, Tn - 1, H, W)
        excess = F.relu(T[:, 1:] - local_max)
        return self.weight * excess.pow(2).mean()


# ---------------------------------------------------------------------------
# 3. Sensor Dirichlet pinning
# ---------------------------------------------------------------------------


class SensorPinningLoss(nn.Module):
    """Hard Dirichlet penalty at the four sensor pixels.

    ``sensors`` is a list of ``(y, x)`` integer coordinates and
    ``values`` is a ``(B, T, N_sensors)`` tensor of measured temperatures.
    """

    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        T: torch.Tensor,
        sensors: "list[tuple[int, int]]",
        values: torch.Tensor,
    ) -> torch.Tensor:
        if not _is_spatial_seq(T):
            return _zero(T)
        loss = _zero(T)
        for i, (y, x) in enumerate(sensors):
            pred = T[:, :, y, x]  # (B, T)
            loss = loss + F.mse_loss(pred, values[..., i])
        return self.weight * loss / max(len(sensors), 1)


# ---------------------------------------------------------------------------
# 4. Advection-diffusion residual using optical-flow as velocity
# ---------------------------------------------------------------------------


class AdvectionDiffusionLoss(nn.Module):
    r"""Residual of :math:`\partial_t T + v\cdot\nabla T - \alpha\nabla^2 T = 0`.

    ``flow`` carries the per-frame optical flow ``(B, T, 2, H, W)`` already
    present in the dataset's last two channels.
    """

    def __init__(self, weight: float = 1.0, alpha: float = 1e-3, dx: float = 1.0, dt: float = 1.0):
        super().__init__()
        self.weight = weight
        self.alpha = alpha
        self.dx = dx
        self.dt = dt

    def forward(self, T: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        if not _is_spatial_seq(T) or T.shape[1] < 2:
            return _zero(T)
        dTdt = (T[:, 1:] - T[:, :-1]) / self.dt
        Tx, Ty = _grad_xy(T[:, :-1], dx=self.dx)
        # flow expected shape: (B, T, 2, H, W) or (B, T-1, 2, H, W)
        vx = flow[:, : Tx.shape[1], 0]
        vy = flow[:, : Ty.shape[1], 1]
        adv = vx * Tx + vy * Ty
        lap = _laplacian(T[:, :-1], dx=self.dx)
        residual = dTdt + adv - self.alpha * lap
        return self.weight * residual.pow(2).mean()


# ---------------------------------------------------------------------------
# 5. Radial symmetry around antenna tip
# ---------------------------------------------------------------------------


class RadialSymmetryLoss(nn.Module):
    """Penalise angular variance of T at constant radius from a centre."""

    def __init__(self, weight: float = 0.1, n_radii: int = 8, n_angles: int = 16):
        super().__init__()
        self.weight = weight
        self.n_radii = n_radii
        self.n_angles = n_angles

    def forward(
        self, T: torch.Tensor, centre: Optional["tuple[int, int]"] = None
    ) -> torch.Tensor:
        if not _is_spatial_seq(T):
            return _zero(T)
        B, Tn, H, W = T.shape
        cy, cx = centre if centre is not None else (H // 2, W // 2)
        ys = torch.arange(H, device=T.device).view(-1, 1) - cy
        xs = torch.arange(W, device=T.device).view(1, -1) - cx
        r = torch.sqrt(ys.float() ** 2 + xs.float() ** 2)
        r_max = float(min(H, W) // 2 - 1)
        bins = torch.linspace(1.0, r_max, self.n_radii, device=T.device)
        loss = _zero(T)
        for rb in bins:
            mask = (r >= rb - 0.5) & (r < rb + 0.5)
            if mask.sum() < 4:
                continue
            vals = T[:, :, mask]  # (B, T, K)
            loss = loss + vals.var(dim=-1).mean()
        return self.weight * loss / max(len(bins), 1)


# ---------------------------------------------------------------------------
# 6. Arrhenius damage / CEM43 surrogate
# ---------------------------------------------------------------------------


class ArrheniusDamageLoss(nn.Module):
    r"""Soft CEM-43 surrogate loss.

    Penalises sequences whose Arrhenius damage integral

    .. math::
        \Omega(\mathbf x) = \int_0^t A\,e^{-E_a/(R T(\mathbf x,\tau))}\,d\tau

    deviates from a target dose map.  Useful when ground-truth ablation
    zones (from cut phantom imaging) are available as a binary mask
    ``target_zone``.  When no target is provided we simply maximise
    contrast between zone and non-zone -- callers should weight this
    carefully.
    """

    def __init__(
        self,
        weight: float = 1.0,
        A: float = 3.1e98,  # Henriques and Moritz collagen denaturation
        Ea: float = 6.28e5,  # J/mol
        R: float = 8.314,  # J/(mol K)
        dt: float = 1.0 / 30,
    ):
        super().__init__()
        self.weight = weight
        self.A = A
        self.Ea = Ea
        self.R = R
        self.dt = dt

    def forward(
        self, T_celsius: torch.Tensor, target_zone: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not _is_spatial_seq(T_celsius):
            return _zero(T_celsius)
        # Convert C -> K and clip to avoid overflow in the exponent.
        T_k = (T_celsius + 273.15).clamp(min=273.15, max=400.0)
        rate = self.A * torch.exp(-self.Ea / (self.R * T_k))
        omega = (rate * self.dt).sum(dim=1)  # (B, H, W)
        damaged = (omega > 1.0).float()
        if target_zone is None:
            # Encourage non-trivial zones (avoid trivial all-zero solutions).
            return self.weight * (damaged.mean() - 0.05).pow(2)
        return self.weight * F.binary_cross_entropy(
            damaged.clamp(1e-6, 1 - 1e-6), target_zone.float()
        )


# ---------------------------------------------------------------------------
# 7. TV / smoothness prior on learned parameter maps
# ---------------------------------------------------------------------------


class ParameterTVLoss(nn.Module):
    """Anisotropic total-variation penalty on a learned spatial parameter map."""

    def __init__(self, weight: float = 1e-3):
        super().__init__()
        self.weight = weight

    def forward(self, param_map: torch.Tensor) -> torch.Tensor:
        if param_map.dim() < 2:
            return _zero(param_map)
        dx = (param_map[..., :, 1:] - param_map[..., :, :-1]).abs().mean()
        dy = (param_map[..., 1:, :] - param_map[..., :-1, :]).abs().mean()
        return self.weight * (dx + dy)


# ---------------------------------------------------------------------------
# convenience aggregator
# ---------------------------------------------------------------------------


class CompositePhysicsLoss(nn.Module):
    """Sum of the additional losses with individual weights.

    Pass weights = 0 to disable a term.  Always returns a scalar tensor.
    """

    def __init__(
        self,
        w_energy: float = 0.0,
        w_max_principle: float = 0.0,
        w_advection: float = 0.0,
        w_radial: float = 0.0,
        w_arrhenius: float = 0.0,
        w_tv: float = 0.0,
        w_sensor: float = 0.0,
        alpha: float = 1e-3,
        dx: float = 1.0,
        dt: float = 1.0 / 30,
    ):
        super().__init__()
        self.energy = EnergyConservationLoss(w_energy)
        self.max_principle = MaximumPrincipleLoss(w_max_principle)
        self.advection = AdvectionDiffusionLoss(w_advection, alpha=alpha, dx=dx, dt=dt)
        self.radial = RadialSymmetryLoss(w_radial)
        self.arrhenius = ArrheniusDamageLoss(w_arrhenius, dt=dt)
        self.tv = ParameterTVLoss(w_tv)
        self.sensor = SensorPinningLoss(w_sensor)

    def forward(
        self,
        T: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
        param_maps: Optional[torch.Tensor] = None,
        target_zone: Optional[torch.Tensor] = None,
        antenna_centre: Optional["tuple[int, int]"] = None,
        sensor_coords: Optional["list[tuple[int, int]]"] = None,
        sensor_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total = _zero(T)
        total = total + self.energy(T)
        total = total + self.max_principle(T)
        if flow is not None:
            total = total + self.advection(T, flow)
        total = total + self.radial(T, antenna_centre)
        total = total + self.arrhenius(T, target_zone)
        if param_maps is not None:
            total = total + self.tv(param_maps)
        if sensor_coords is not None and sensor_values is not None:
            total = total + self.sensor(T, sensor_coords, sensor_values)
        return total


__all__ = [
    "EnergyConservationLoss",
    "MaximumPrincipleLoss",
    "SensorPinningLoss",
    "AdvectionDiffusionLoss",
    "RadialSymmetryLoss",
    "ArrheniusDamageLoss",
    "ParameterTVLoss",
    "CompositePhysicsLoss",
]
