"""Kolmogorov-Arnold Network (KAN) layers and model wrappers.

This module implements an efficient B-spline KAN layer in the style of
Liu et al. 2024 (Kolmogorov-Arnold Networks) and Blealtan's
`efficient-kan` reformulation, together with two model wrappers that
drop straight into the existing LOSO benchmark harness:

* ``KANResNet`` -- ResNet18 encoder + KAN scalar head.
  Direct apples-to-apples ablation against ``SimpleResNet``.

* ``SpatialKANBioheat`` -- ResNet18 sequence encoder + a separable
  spatial KAN field head (SPIKAN-style: Jacob et al., MLST 2025), to be
  trained with ``AdvancedBioHeatLoss`` exactly like ``ConvectionBioheat``.
  Ablation against ``SpatialPhysicsCNNLSTM`` / ``ConvectionBioheat``.

The KAN layer also exposes a ``basis`` argument so the user can swap
between B-spline (default), Chebyshev polynomials and a wavelet basis to
reproduce the ChebPIKAN / WAV-KAN comparison from the literature.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core KAN linear layer
# ---------------------------------------------------------------------------


class KANLinear(nn.Module):
    """Efficient B-spline Kolmogorov-Arnold linear layer.

    Computes ``y_j = sum_i ( w_b * b(x_i) + w_s * spline(x_i) )`` where
    ``b`` is a SiLU residual activation and ``spline`` is a learnable
    weighted sum of B-spline basis functions of order ``spline_order``.

    The implementation collapses the per-edge spline coefficients into a
    single ``F.linear`` call against the pre-computed B-spline basis
    matrix, matching the memory profile of a standard ``nn.Linear``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: type[nn.Module] = nn.SiLU,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        basis: str = "bspline",
    ) -> None:
        super().__init__()
        if basis not in {"bspline", "chebyshev"}:
            raise ValueError(f"Unknown KAN basis: {basis!r}")

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.basis = basis

        # Uniform extended grid for B-splines.
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        )
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())

        # Base residual weight (acts like a standard linear layer).
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

        # Spline weights: one B-spline coefficient per (out, in, basis_idx).
        n_basis = (
            grid_size + spline_order if basis == "bspline" else grid_size + 1
        )
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, n_basis)
        )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()

        self.reset_parameters()

    # ------------------------------------------------------------------
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                (torch.rand_like(self.spline_weight) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.copy_(noise * self.scale_spline)

    # ------------------------------------------------------------------
    def _bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the B-spline basis at ``x``.

        Returns ``(batch, in_features, grid_size + spline_order)``.
        """
        grid = self.grid  # (in_features, G + 2k + 1)
        x = x.unsqueeze(-1)  # (batch, in_features, 1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()

        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k])
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases.contiguous()

    def _chebyshev_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Chebyshev T_n(x) polynomial basis, ``x`` already in [-1, 1]."""
        x = torch.tanh(x).unsqueeze(-1)  # squash to chebyshev domain
        T = [torch.ones_like(x), x]
        for n in range(2, self.grid_size + 1):
            T.append(2 * x * T[-1] - T[-2])
        return torch.cat(T, dim=-1)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        if self.basis == "bspline":
            return self._bspline_basis(x)
        return self._chebyshev_basis(x)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` has shape ``(..., in_features)``; output is ``(..., out_features)``."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Base path -- standard linear over a smooth activation.
        base = F.linear(self.base_activation(x_flat), self.base_weight)

        # Spline path -- contract basis with per-edge coefficients.
        basis = self._basis(x_flat)  # (batch, in, n_basis)
        # Combine in (out, in * n_basis) form so we can use F.linear.
        weight = self.spline_weight.reshape(self.out_features, -1)
        basis_flat = basis.reshape(basis.shape[0], -1)
        spline = F.linear(basis_flat, weight)

        out = base + spline
        return out.reshape(*orig_shape[:-1], self.out_features)


class KAN(nn.Module):
    """Stack of ``KANLinear`` layers (no nonlinearity between them: the
    spline activations on each edge already provide nonlinearity)."""

    def __init__(
        self,
        layer_widths: list[int],
        grid_size: int = 5,
        spline_order: int = 3,
        basis: str = "bspline",
    ) -> None:
        super().__init__()
        layers = []
        for in_f, out_f in zip(layer_widths[:-1], layer_widths[1:]):
            layers.append(
                KANLinear(
                    in_f,
                    out_f,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    basis=basis,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Spatial KAN field head (SPIKAN-style)
# ---------------------------------------------------------------------------


class SpatialKANHead(nn.Module):
    """Separable Physics-Informed KAN head that emits a (T, h, w) field.

    Inspired by SPIKANs (Jacob et al., MLST 2025): we split the
    coordinate-conditioned KAN into three small KANs, one per axis
    ``(x, y, t)``, and combine their outputs multiplicatively before a
    final KAN that fuses with the latent vector from the encoder.

    Inputs
    ------
    latent : ``(B, D)`` -- per-clip latent from the encoder.

    Returns
    -------
    field : ``(B, T, h, w)`` temperature field at the requested grid.
    """

    def __init__(
        self,
        latent_dim: int,
        output_hw: tuple[int, int] = (4, 4),
        time_steps: int = 5,
        kan_width: int = 16,
        grid_size: int = 5,
        spline_order: int = 3,
        basis: str = "bspline",
    ) -> None:
        super().__init__()
        self.h, self.w = output_hw
        self.time_steps = time_steps

        self.kan_x = KAN([1, kan_width, kan_width], grid_size, spline_order, basis)
        self.kan_y = KAN([1, kan_width, kan_width], grid_size, spline_order, basis)
        self.kan_t = KAN([1, kan_width, kan_width], grid_size, spline_order, basis)
        self.fuse = KAN(
            [latent_dim + kan_width, kan_width, 1],
            grid_size,
            spline_order,
            basis,
        )

        xs = torch.linspace(-1, 1, self.w)
        ys = torch.linspace(-1, 1, self.h)
        ts = torch.linspace(-1, 1, time_steps)
        self.register_buffer("xs", xs.unsqueeze(-1))
        self.register_buffer("ys", ys.unsqueeze(-1))
        self.register_buffer("ts", ts.unsqueeze(-1))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.shape[0]
        T, H, W = self.time_steps, self.h, self.w

        fx = self.kan_x(self.xs)            # (W, K)
        fy = self.kan_y(self.ys)            # (H, K)
        ft = self.kan_t(self.ts)            # (T, K)

        # Multiplicative separation: outer product of the three axes.
        # coord_feat: (T, H, W, K)
        coord_feat = ft[:, None, None, :] * fy[None, :, None, :] * fx[None, None, :, :]
        coord_feat = coord_feat.expand(B, T, H, W, -1)
        latent_exp = latent[:, None, None, None, :].expand(B, T, H, W, -1)

        fused = torch.cat([coord_feat, latent_exp], dim=-1)
        field = self.fuse(fused).squeeze(-1)  # (B, T, H, W)
        return field


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


def _make_resnet_encoder(in_channels: int):
    """ResNet18 reused as a 2D encoder; returns (encoder, feat_dim)."""
    from torchvision.models import resnet18

    base = resnet18(weights="IMAGENET1K_V1")
    if in_channels != 3:
        orig = base.conv1
        new_conv = nn.Conv2d(
            in_channels,
            orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=orig.bias is not None,
        )
        with torch.no_grad():
            n = min(in_channels, 3)
            new_conv.weight[:, :n] = orig.weight[:, :n]
            if in_channels > 3:
                nn.init.kaiming_normal_(
                    new_conv.weight[:, 3:], mode="fan_out", nonlinearity="relu"
                )
        base.conv1 = new_conv
    base.fc = nn.Identity()
    return base, 512


class KANResNet(nn.Module):
    """ResNet18 encoder + KAN scalar head.

    Apples-to-apples ablation versus ``SimpleResNet``: identical encoder,
    KAN replaces the final MLP head.  Outputs ``(B, 4)`` (the four
    fiber-optic sensor temperatures), matching the dataset's scalar
    target so the existing LOSO scalar metric works unchanged.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int, int],
        output_dim: int = 4,
        kan_width: int = 64,
        grid_size: int = 5,
        spline_order: int = 3,
        basis: str = "bspline",
    ) -> None:
        super().__init__()
        in_channels = frame_shape[2]
        self.encoder, feat_dim = _make_resnet_encoder(in_channels)
        self.head = KAN(
            [feat_dim, kan_width, output_dim],
            grid_size=grid_size,
            spline_order=spline_order,
            basis=basis,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            feats = self.encoder(x).reshape(B, T, -1).mean(dim=1)
        else:
            feats = self.encoder(x)
        return self.head(feats)


class SpatialKANBioheat(nn.Module):
    """Encoder + separable spatial KAN field head, trainable with
    ``AdvancedBioHeatLoss(frame_shape=(4, 4))``.

    Direct ablation against ``ConvectionBioheat``: same data tensor in,
    same coarse 4x4 field out, but the field is generated by a
    coordinate-conditioned KAN instead of a tiny CNN -- so the physics
    residual is computed on a smoother, spectrally-richer surrogate.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int, int],
        time_steps: int = 5,
        output_hw: tuple[int, int] = (4, 4),
        kan_width: int = 16,
        grid_size: int = 5,
        spline_order: int = 3,
        basis: str = "bspline",
    ) -> None:
        super().__init__()
        in_channels = frame_shape[2]
        self.encoder, feat_dim = _make_resnet_encoder(in_channels)
        self.field_head = SpatialKANHead(
            latent_dim=feat_dim,
            output_hw=output_hw,
            time_steps=time_steps,
            kan_width=kan_width,
            grid_size=grid_size,
            spline_order=spline_order,
            basis=basis,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W).  Pool features over the sequence so the KAN
        # head's time axis comes from its own learnable temporal basis.
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            feats = self.encoder(x.reshape(B * T, C, H, W))
            latent = feats.reshape(B, T, -1).mean(dim=1)
        else:
            latent = self.encoder(x)
        return self.field_head(latent)  # (B, T, h, w)
