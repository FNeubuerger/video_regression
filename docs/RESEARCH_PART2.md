# Part 2: Research & Advanced Architectures

## 1. Physics Priors for Temperature Estimation

**Problem:** We are training a dense temperature map estimator using only sparse supervision (4 points). The model has to "guess" the distribution in the rest of the image.

**Idea:** We have strong *a priori* knowledge about the physics of the setup:
1.  **Heat Source:** localized at the center (the "bright wide line" probe interface).
2.  **Diffusion:** Heat diffuses outwards from this source.
3.  **Distribution:** The temperature field is likely Gaussian-like or decaying from the center.

** Proposed Strategy: The "Physics Prior" Map **

Instead of asking the network to learn $f(I) = T$, we can formulate the problem as learning a residual correction to a known physical prior.

### Formulation A: Input Injection
Append a "Prior Channel" $P$ to the input image.
-   $Input = [R, G, B, P]$
-   Where $P(x,y)$ is a static (or power-dependent) Gaussian map centered on the Active Zone.
-   The network learns to use $P$ as a baseline feature.

### Formulation B: Residual Learning (Preferred)
The model predicts the *deviation* from the prior.
$$ T_{pred}(x,y) = T_{prior}(x,y) + \Delta T_{net}(x,y) $$

-   **$T_{prior}$**: A pre-calculated Gaussian map based on the known Probe Power ($W$) and Active Zone location.
    -   $T_{prior} \propto W \cdot \exp(-\frac{(x-c_x)^2 + (y-c_y)^2}{2\sigma^2})$
-   **$\Delta T_{net}$**: The U-Net output.
-   **Benefit:** The model starts with a "physically plausible" guess (the Prior). It only needs to learn the *local variations* caused by tissue heterogeneity, rather than learning the global diffusion pattern from scratch.

### Benchmarking Plan
We will compare three approaches:
1.  **Baseline (Flat Prior):** Standard U-Net, zero initialization.
2.  **Input Prior:** U-Net with Prior Channel.
3.  **Residual Prior:** $T = T_{prior} + U\text{-}Net(I)$.

---

## 2. Liquid Time Constant (LTC) Networks & Neural Circuit Policies (NCPs)

**Context:** The user inquired about using [LTCs/NCPs](https://github.com/mlech26l/ncps) for the PDE/Loss part.

**Analysis:**
LTCs (Liquid Time-Constant networks) are a subclass of Continuous-Time RNNs (CT-RNNs). unlike standard RNNs/LSTMs which follow discrete update steps, LTCs model the hidden state $x(t)$ using an Ordinary Differential Equation (ODE):

$$ \frac{dx(t)}{dt} = - \left[ \frac{1}{\tau_{sys}} + f(I(t)) \right] \cdot x(t) + f(I(t)) \cdot A $$

Where the time-constant $\tau$ is input-dependent ("liquid"). This formulation is explicitly inspired by biological synapses and RC circuits.

**Relevance to Bioheat Transfer:**
The Bioheat Transfer Equation is also a differential equation with a decay term:
$$ \rho c \frac{\partial T}{\partial t} = \nabla (k \nabla T) - w_b c_b (T - T_a) + Q $$
The perfusion term $-w_b c_b (T - T_a)$ is mathematically analogous to the leakage term in the LTC formulation. By using LTCs, we hypothesize the network can learn the *continuous-time dynamics* of tissue heating more naturally than discrete RNNs, handling irregular sampling rates and providing better stability.

### Implementation: Latent-LTC
We implemented a hybrid architecture `LatentLTC_UNet` in `models/latent_ltc.py`:
1.  **Encoder (ResNet):** Compresses the high-dimensional frame $I_t$ into a compact latent vector $z_t$.
2.  **Dynamics (LTC/NCP):** The latent state evolves over time according to the LTC ODEs. This models the physics in a reduced dimensionality.
3.  **Decoder (U-Net):** The evolved latent state is decoded back into the dense temperature map.

This enables us to model the *physics* in the latent space (fast, temporal) while maintaining high-resolution spatial outputs.

---
The Bioheat Transfer Equation (BHTE) is a PDE describing diffusion:
$$ \rho c \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + Q_{source} - Q_{perfusion} $$

This maps conceptually well to LTCs:
1.  **Continuous Time:** BHTE describes a continuous evolution. LTCs solve an ODE solver (e.g., explicit RK4 or semi-implicit), allowing them to handle irregular time sampling or "stiff" dynamics better than LSTMs.
2.  **Decay/Perfusion:** The term $-Q_{perfusion}$ (blood flow cooling) acts as a leakage term, similar to the $-\frac{1}{\tau} x(t)$ term in LTCs.
3.  **Stability:** LTCs have bounded dynamics (Bounded Input Bounded Output), preventing the "exploding gradient" or unbounded predictions that can occur in physics-informed unconstrained networks.

**Architecture: Neural Circuit Policies (NCPs)**
NCPs are a specific sparse wiring architecture for LTCs (Sensors $\to$ Interneurons $\to$ Command $\to$ Motor). They significantly reduce parameter count and improve interpretability compared to fully connected RNNs.

## 3. Artifact Mitigation & Spatial Masking

**Problem:** Non-tissue objects (antennas, thermometers) appear as bright artifacts in US video. Models tend to "cheat" by learning shortcut correlations between artifact brightness and temperature labels, rather than learning the subtle tissue texture changes.

**Solution: Automated Artifact Masking**
We implemented a multi-stage spatial masking strategy:
1.  **Sensor Masking:** Uses `sensor_coordinates.json` to identify the $9 \times 9$ pixel regions around each thermometer.
2.  **Antenna Detection:** An automated detector uses the **Hough Line Transform** on Canny edges to identify the linear antenna structure in every frame.
3.  **Input Suppression:** The identified regions are zerod-out in the input frame before being passed to the model.
4.  **Physics Residual Masking:** The spatial loss residuals (e.g., Laplacian of temperature) are multiplied by $(1 - Mask)$ to ensure the model is only penalized for physics violations in pure tissue regions.

**Experimental Setup:**
We compare "Masked" vs "Unmasked" training runs across all architectures to quantify the degree of shortcut learning. Successful generalization to masked frames is a requirement for clinical relevance.

### Implementation Plan (Optional Future Work)

To leverage LTCs for Spatiotemporal Temperature Estimation, we cannot simply use the `ncps` library "out of the box" because it expects 1D vector inputs. We propose two architectures:

#### Architecture A: Latent-Space Dynamics (Feasible)
Model the "global" physics logic in a compressed latent space.

1.  **Encoder (CNN):** Compresses Frame $V_t (H \times W)$ into a latent vector $z_t \in \mathbb{R}^{128}$.
2.  **Dynamics (LTC/NCP):** The LTC layer updates the latent state over time:
    *   Input: $z_t$
    *   State: $h_t$
    *   Output: $h_{t+1}$ (representing the evolved abstract heat state).
3.  **Decoder (CNN):** Upsamples $h_{t+1}$ to the predicted dense map $\hat{T}_{t+1} (H \times W)$.

*Status:* High feasibility. Can use `ncps.wiring.AutoNCP` and `ncps.torch.LTC`.

#### Architecture B: Convolutional LTC (High Risk / High Reward)
Truly Spatiotemporal ODE learning. Defines a "Liquid Cell" at every pixel, coupled spatially by convolutions.

*   **Logic:**
    $$ \frac{dX_{h,w}}{dt} = - \left[ \frac{1}{\tau} + \text{Conv}(X) \right] \cdot X_{h,w} + \text{Conv}(Input) $$
*   **Implementation:** Requires writing a custom `nn.Module` that implements the semi-implicit Euler solver but replaces dense matrix multiplications with `Conv2d` operations.
*   **Benefit:** Directly learns the diffusion kernel $\nabla \cdot (k \nabla T)$.

### Proposed Experiment Roadmap
If we proceed with Issue #23:
1.  **Repository:** Clone/Install `ncps` (`pip install ncps`).
2.  **Prototype:** Create `models/latent_ltc.py`.
    *   Class `LatentLTC(nn.Module)`:
        *   `self.encoder = ResNetEncoder(...)`
        *   `self.rnn = LTC(input_size=128, units=32, wiring=AutoNCP(32, 1))`
        *   `self.decoder = UNetDecoder(...)`
3.  **Baseline Comparison:** Compare `LatentLTC` vs `ConvLSTM` on the `sequence_1` data (time-series).
    *   Metric: Generalization to unseen time-steps (extrapolation).

**Recommendation:**
Prioritize **Physics Priors (Spatial)** first (Issue #24). LTCs address *temporal* dynamics, which is Part 3 of our roadmap. Keep this research on hold until spatial estimation is solid.
