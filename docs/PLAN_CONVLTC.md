# Plan: Convolutional Liquid Time Constant (ConvLTC) Network

## Objective
Implement a spatially-aware continuous-time recurrent neural network. Unlike `LatentLTC` which flattens spatial structure into a latent vector, `ConvLTC` maintains $(H, W)$ dimensions throughout the temporal evolution. This aligns better with the Bioheat Equation where diffusion is a local spatial process.

## Theoretical Formulation
The hidden state $H_t$ is a 3D tensor $(C, H, W)$. The dynamics follow the LTC ODE:
$$ \frac{dH}{dt} = - \left[ \frac{1}{\tau} + f(I_t, H_t) \right] \odot H_t + g(I_t, H_t) $$

where:
*   $\odot$ is element-wise multiplication.
*   $f(\cdot)$ (Decay/Leakage) and $g(\cdot)$ (Drive/Input) are parameterised by **Convolutional Neural Networks** instead of dense layers.
*   This effectively learns a "Liquid Pixel" at every spatial location, coupled to its neighbors via the convolution kernel.

## Architecture Design

### 1. ConvLTC Cell (`models/conv_ltc.py`)
*   **State:** `h` (Hidden), `c` (Cell? No, LTC is usually just `h`. But generic ODE solvers might not be needed if we use Euler discretization).
*   **ODE Solver:** Explicit Euler (simple and fast) or a Semi-Implicit method if stiff.
*   **Forward Step:**
    1.  Concatenate Input $X_t$ and State $H_{t-1}$.
    2.  Compute $G = \text{Sigmoid}(Conv(X, H))$ (Input-dependent Decay rate).
    3.  Compute $S = \text{Tanh}(Conv(X, H))$ (Input-dependent Drive).
    4.  Update: $H_t = H_{t-1} + \Delta t \cdot (-(\text{base\_leak} + G) \cdot H_{t-1} + S)$.

### 2. High-Level Model (`ConvLTC_Model`)
*   **Encoder:** Shallow ResNet or Simple CNN to extract features $F_t$ from Image $I_t$.
    *   Input: $(B, T, 3, H, W) \to (B, T, C_{feat}, H, W)$.
*   **Temporal Dynamics:** Recurrent loop using `ConvLTCCell`.
    *   Input: $F_t$.
    *   State: $H_t$.
*   **Decoder:** 1x1 Convolution to map $H_t \to \text{TempMap}_t$.

## Implementation Steps

1.  [x] Define Plan.
2.  [ ] Implement `ConvLTCCell` and `ConvLTC` sequence processor in `models/conv_ltc.py`.
3.  [ ] Update `training/train_ltc.py` to support `--model_type conv_ltc`.
4.  [ ] Add Unit Tests in `tests/test_model_suite.py`.
5.  [ ] Launch Benchmark.

## Risk Mitigation
*   **Memory:** ConvLTC stores dense states for BPTT. If sequence is long, GPU memory explodes.
    *   *Mitigation:* Use smaller `sequence_length` (e.g., 8 or 16), small state channels (e.g., 32), or gradient checkpointing.
*   **Stability:** ODEs can be unstable.
    *   *Mitigation:* Bounded activations (Sigmoid/Tanh) and small $\Delta t$.

