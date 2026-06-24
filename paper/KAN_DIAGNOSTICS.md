"""Diagnostic analysis of KAN basis functions and failure modes.

This document provides insight into why KANResNet failed (3358 MAE) vs
SpatialKANBioheat (47.8 MAE), and the interpretability advantage of KANs.
"""

# KAN Interpretability & Failure Analysis

## KAN Basis Function Representation

Kolmogorov-Arnold Networks (KANs) offer a key interpretability advantage:
instead of opaque nonlinearities like ReLU, the learned activation
functions can be directly visualized and analyzed. Each KANLinear layer
decomposes as:

    y_j = Σ_i [ w_b,ji * σ(x_i) + w_s,ji * Σ_k c_k * B_k(x_i) ]

where σ is a base activation (SiLU), B_k are B-spline basis functions,
and the coefficients c are trainable. The paper/figures/kan_basis_*.png
plots show these learned univariate transformations for both models.

## KANResNet Failure Diagnosis

**Problem**: KANResNet achieves 3358 K MAE, worse than any baseline or
even naive constant prediction. Per-fold breakdown reveals severe overfitting:
- US_001: 65.3 K (reasonable)
- US_002: 1321.3 K (exploding)
- US_006: 6811.5 K (catastrophic)
- US_013: 16034.0 K (catastrophic)
- US_015: 23091.7 K (catastrophic)

**Root causes** (identified via basis function visualization):
1. **Unconstrained spline growth**: The spline coefficients can grow
   arbitrarily large with no regularization, causing basis functions to
   become highly oscillatory and nonsmooth.
2. **Overfitting to individual subjects**: With 512-element bottleneck
   and small per-fold training sets (~1300 sequences), the scalar KAN
   head overfits catastrophically to low-entropy subjects.
3. **No physics bias**: Unlike SpatialKANBioheat (which has the
   AdvancedBioHeatLoss enforcing smooth spatio-temporal fields),
   KANResNet has only MSE and no inductive bias.

**Basis function signature**: Plots show extreme oscillations, high
dynamic range (max ~50–100), and no smooth structure. These are
indicators of Gibbs phenomenon / basis function aliasing.

## SpatialKANBioheat Success

**Result**: 47.8 K MAE, competitive with ConvectionBioheat (49.9 K).

**Why it works**:
1. **Physics-informed loss**: AdvancedBioHeatLoss constrains the spatio-
   temporal field to obey energy conservation and limits field magnitude.
2. **Separable structure**: The SPIKAN-style factorization (independent
   1D KANs per (x, y, t) coordinate) reduces degrees of freedom and
   overfitting surface.
3. **Multihead output**: The (T, h, w) field provides redundancy and
   distributed target signal, not a single scalar.

**Basis function signature**: Still shows moderate oscillations (max
~50–70), but the physics loss prevents pathological growth.

## Recommendations for Future Work

1. **KAN regularization**: L2 penalty on spline weights or adaptive
   grid refinement (ChebPIKAN, adaptive-PIKAN) to prevent blow-up.
2. **Batch normalization**: Add normalization between KAN layers to
   stabilize training dynamics.
3. **Ensemble masking**: Apply per-subject dropout or a gating network
   to prevent overfitting to high-variance subjects (e.g., US_002).
4. **Better initialization**: Use spectral initialization or knowledge
   distillation from a pre-trained SimpleResNet.

## Conclusion

KANs' interpretability (via visualizable basis functions) is a genuine
advantage for scientific ML. However, their flexibility requires careful
regularization—the LOSO results show that a physics-informed loss
(SpatialKANBioheat) succeeded, while an unregularized scalar head
(KANResNet) failed spectacularly. Future work should apply the insights
from PINN literature (Adaptive-PIKAN, ChebPIKAN, RAD-PIKAN) to stabilize
data-driven KAN training.
