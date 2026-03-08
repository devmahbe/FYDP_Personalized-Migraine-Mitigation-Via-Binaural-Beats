# Adaptive Control Law — Detailed Mathematical Explanation

## Weighted EEG Feedback Controller with Online Learning for Binaural Beat Therapy

**Reference Notebook:** `adaptive_migraine_treatment.ipynb`

---

## 1. The Equations

The adaptive system in the notebook uses two coupled equations that run every update interval (60 seconds):

### Equation 1 — Frequency Update

$$f_b(t+\Delta t) = f_b(t) + \alpha \cdot \left(\sum_{k} w_k(t) \cdot \Delta P_k(t)\right) + \beta \cdot E(t)$$

**With hard constraint:**

$$f_b(t+\Delta t) = \text{clip}\big(f_b(t+\Delta t),\ f_{\min} = 4\ \text{Hz},\ f_{\max} = 13\ \text{Hz}\big)$$

### Equation 2 — Weight Update (Online Learning)

$$w_k(t+\Delta t) = w_k(t) + \eta \cdot \big(P_{k,\text{target}} - P_{k,\text{observed}}(t)\big)$$

**With constraint:**

$$w_k \in [0.1,\ 2.0]$$

### Alternative Simple Control Law (Not Used)

$$f_b(t+\Delta t) = f_b(t) + K \cdot (\text{target alpha} - \text{observed alpha})$$

This simplified version was documented as an alternative but is not implemented in the notebook.

---

## 2. Symbol Definitions

### Frequency Update Symbols

| Symbol          | Name                          | Default Value | Description                                                                 |
|-----------------|-------------------------------|---------------|-----------------------------------------------------------------------------|
| $f_b(t)$        | Beat frequency at time $t$    | —             | Current binaural beat frequency in Hz                                       |
| $f_b(t+\Delta t)$ | Updated beat frequency     | —             | Frequency after one control step                                            |
| $\alpha$        | EEG feedback learning rate    | 0.05          | Controls how strongly EEG band power changes affect frequency               |
| $\beta$         | Clinical feedback weight      | 0.10          | Controls how strongly patient feedback affects frequency                    |
| $w_k(t)$        | Weight for band $k$ at time $t$ | varies     | Adaptive weight reflecting importance of each EEG band                      |
| $\Delta P_k(t)$ | Band power change             | —             | Change in normalized power for band $k$: $P_k(t+\Delta t) - P_k(t)$       |
| $E(t)$          | Clinical feedback score       | ∈ [-1, +1]    | Patient's subjective improvement rating at time $t$                         |
| $k$             | Band index                    | —             | Iterates over: delta, theta, alpha, beta, gamma                             |
| $f_{\min}$      | Minimum frequency             | 4 Hz          | Lower bound (theta range)                                                   |
| $f_{\max}$      | Maximum frequency             | 13 Hz         | Upper bound (top of alpha range)                                            |

### Weight Update Symbols

| Symbol               | Name                          | Default Value | Description                                                          |
|----------------------|-------------------------------|---------------|----------------------------------------------------------------------|
| $\eta$               | Weight learning rate          | 0.03          | Controls how fast weights adapt to errors                            |
| $P_{k,\text{target}}$ | Target power for band $k$   | varies        | Desired normalized power ratio (therapeutic goal)                    |
| $P_{k,\text{observed}}(t)$ | Observed power at time $t$ | —          | Measured normalized power ratio from EEG                             |

### Target EEG State Values

| Band    | $P_{k,\text{target}}$ | Therapeutic Rationale                         |
|---------|------------------------|-----------------------------------------------|
| Delta   | 0.15 (15%)             | Low — avoid excessive drowsiness              |
| Theta   | 0.20 (20%)             | Moderate — promote relaxation                 |
| Alpha   | 0.40 (40%)             | **HIGH — primary therapeutic target**         |
| Beta    | 0.20 (20%)             | Moderate — maintain calm alertness            |
| Gamma   | 0.05 (5%)              | Low — reduce cortical hyperexcitability       |

---

## 3. How It Works — Step by Step

### Overview: The Treatment Loop

The system runs for 20 minutes with updates every 60 seconds (20 total steps). At each step:

```
1. Deliver binaural beat at current frequency for 60 seconds
2. Measure new EEG state (band powers)
3. Calculate band power changes (ΔP_k)
4. Collect clinical feedback score (E(t))
5. Update weights using Equation 2
6. Update frequency using Equation 1
7. Repeat
```

### Step 1: Measure EEG Band Power Changes

At each update, the system computes the change in normalized power for each of the five bands:

$$\Delta P_k(t) = P_k(t + \Delta t) - P_k(t)$$

For example, if alpha power was 25% at minute 3 and is now 28% at minute 4:

$$\Delta P_{\text{alpha}}(3) = 0.28 - 0.25 = +0.03$$

A positive $\Delta P_k$ means that band's power increased; negative means it decreased.

### Step 2: Compute Weighted EEG Signal

The five band power changes are combined into a single scalar using the adaptive weights:

$$\text{weighted\_eeg\_change} = \sum_{k \in \{\delta, \theta, \alpha, \beta, \gamma\}} w_k(t) \cdot \Delta P_k(t)$$

The weights $w_k$ determine how much influence each band's change has on the frequency update. Bands that are further from their target get higher weights (via Equation 2), so the controller pays more attention to the most "off-target" bands.

### Step 3: Incorporate Clinical Feedback

The clinical score $E(t) \in [-1, +1]$ represents the patient's subjective state:

| Score Range   | Interpretation                                   |
|---------------|--------------------------------------------------|
| $E(t) < 0$   | Patient reports worsening (pain, discomfort)     |
| $E(t) = 0$   | No change                                        |
| $E(t) > 0$   | Patient reports improvement (pain relief)        |

In the simulation, this is modeled as:
- Early treatment ($t < 3$ min): slight discomfort ($E \approx -0.2$)
- Mid treatment ($3 \leq t < 10$ min): gradual improvement ($E \approx 0.3$ to $0.7$)
- Later treatment ($t \geq 10$ min): plateau ($E \approx 0.7$)

### Step 4: Apply the Frequency Update (Equation 1)

The two signals are combined:

$$f_b(t+\Delta t) = f_b(t) + \underbrace{\alpha \cdot \left(\sum_k w_k \cdot \Delta P_k\right)}_{\text{EEG-driven component}} + \underbrace{\beta \cdot E(t)}_{\text{Clinical component}}$$

- **EEG component** ($\alpha = 0.05$): Adjusts frequency based on how the brain's band powers are changing. If the weighted sum is positive (bands are moving in a favorable direction), frequency increases; if negative, it decreases.
- **Clinical component** ($\beta = 0.10$): If the patient reports improvement ($E > 0$), frequency nudges up (reinforcing the current direction). If the patient reports worsening ($E < 0$), frequency nudges down.

The result is clipped to $[4, 13]$ Hz.

### Step 5: Update Weights (Equation 2)

After computing the new EEG state, each band's weight is updated based on how far that band is from its target:

$$w_k(t+\Delta t) = w_k(t) + \eta \cdot \big(P_{k,\text{target}} - P_{k,\text{observed}}(t)\big)$$

- If $P_{k,\text{observed}} < P_{k,\text{target}}$: the error is positive → weight **increases** → this band gets more influence on future frequency updates
- If $P_{k,\text{observed}} > P_{k,\text{target}}$: the error is negative → weight **decreases** → this band gets less influence
- If $P_{k,\text{observed}} = P_{k,\text{target}}$: no change

Weights are clipped to $[0.1, 2.0]$ to prevent any single band from dominating or being ignored.

---

## 4. Numerical Walkthrough

### Example A: Early Treatment (Minute 2)

```
Given:
  f_b(t) = 10.0 Hz
  α = 0.05,  β = 0.10,  η = 0.03

  Current EEG:   delta=0.22, theta=0.18, alpha=0.28, beta=0.24, gamma=0.08
  Previous EEG:  delta=0.23, theta=0.17, alpha=0.26, beta=0.25, gamma=0.09

  Weights: w_delta=0.43, w_theta=0.52, w_alpha=0.62, w_beta=0.46, w_gamma=0.47

  Clinical score: E(t) = -0.15  (early discomfort)

Step 1 — Band power changes:
  ΔP_delta = 0.22 - 0.23 = -0.01
  ΔP_theta = 0.18 - 0.17 = +0.01
  ΔP_alpha = 0.28 - 0.26 = +0.02
  ΔP_beta  = 0.24 - 0.25 = -0.01
  ΔP_gamma = 0.08 - 0.09 = -0.01

Step 2 — Weighted EEG change:
  = (0.43 × -0.01) + (0.52 × 0.01) + (0.62 × 0.02) + (0.46 × -0.01) + (0.47 × -0.01)
  = -0.0043 + 0.0052 + 0.0124 - 0.0046 - 0.0047
  = +0.0040

Step 3 — Frequency update:
  f_b(t+Δt) = 10.0 + 0.05 × 0.0040 + 0.10 × (-0.15)
            = 10.0 + 0.0002 - 0.015
            = 9.9852 Hz

→ Frequency decreases slightly (clinical discomfort outweighs small EEG improvement)
```

### Example B: Mid Treatment (Minute 8, Things Improving)

```
Given:
  f_b(t) = 10.05 Hz
  Weights have adapted: w_alpha = 0.85 (high — alpha is most "needed")

  ΔP_alpha = +0.03  (alpha rising nicely)
  E(t) = +0.55  (patient reports clear improvement)

  Weighted EEG change ≈ 0.85 × 0.03 + (small other terms) ≈ 0.026

  f_b(t+Δt) = 10.05 + 0.05 × 0.026 + 0.10 × 0.55
            = 10.05 + 0.0013 + 0.055
            = 10.106 Hz

→ Frequency increases — reinforcing the therapeutic direction
```

### Example C: Weight Update for Alpha Band

```
Given:
  w_alpha(t) = 0.62,  η = 0.03
  P_alpha_target = 0.40,  P_alpha_observed = 0.28

  w_alpha(t+Δt) = 0.62 + 0.03 × (0.40 - 0.28)
                = 0.62 + 0.03 × 0.12
                = 0.62 + 0.0036
                = 0.6236

→ Alpha weight increases because alpha is below target
→ Future frequency updates will be more responsive to alpha changes
```

### Example D: Weight Update for Gamma Band

```
Given:
  w_gamma(t) = 0.47,  η = 0.03
  P_gamma_target = 0.05,  P_gamma_observed = 0.08

  w_gamma(t+Δt) = 0.47 + 0.03 × (0.05 - 0.08)
                = 0.47 + 0.03 × (-0.03)
                = 0.47 - 0.0009
                = 0.4691

→ Gamma weight decreases because gamma is above target
→ Gamma changes will have less influence on frequency updates
```

---

## 5. Weight Initialization

Weights are not all set to the same starting value. They are initialized based on the patient's **baseline deviation** from the target state and **clinical profile**:

### EEG Band Weights

$$w_k(0) = 0.5 + \big(P_{k,\text{target}} - P_{k,\text{baseline}}\big)$$

If a band is far below its target, $w_k(0) > 0.5$ (more attention). If it's above target, $w_k(0) < 0.5$.

### Clinical Factor Weights (Static)

These weights do not change during the session:

| Factor        | Formula                                                    | Purpose                                    |
|---------------|------------------------------------------------------------|--------------------------------------------|
| $w_{\text{age}}$     | $1.0 - \frac{\text{age} - 20}{60}$, clipped to $[0.3, 1.0]$ | Younger patients respond more to entrainment |
| $w_{\text{gender}}$  | 0.9 (Female) or 0.7 (Male)                               | Females have higher migraine prevalence     |
| $w_{\text{migraine type}}$ | 1.2 (Aura) or 0.8 (Non-Aura)                      | Aura patients need more aggressive therapy  |

---

## 6. Initial Frequency Selection

The starting frequency before the loop begins is computed as:

$$f_b(0) = f_{\text{base}} + \big(P_{\alpha,\text{target}} - P_{\alpha,\text{baseline}}\big) \times 5.0$$

Where:
- $f_{\text{base}} = 10.0$ Hz for **aura** migraine (target alpha band directly)
- $f_{\text{base}} = 7.5$ Hz for **non-aura** migraine (theta-alpha transition)
- The adjustment factor (×5.0) scales the alpha deficit into a Hz offset

Result is clipped to $[4, 13]$ Hz.

**Intuition:** If the patient has very low alpha power (large deficit), the starting frequency is pushed higher into the alpha range to stimulate it. If alpha is already reasonable, the frequency starts lower in the theta-alpha border.

---

## 7. The Dual-Signal Design

The frequency update incorporates **two independent feedback signals**:

```
                    ┌──────────────────────────────┐
  EEG Headset ───→  │  Band Power Extraction       │
                    │  δ, θ, α, β, γ proportions   │
                    └──────────┬───────────────────┘
                               │  ΔP_k(t) for each band
                               ▼
                    ┌──────────────────────────────┐
                    │  Weighted Sum                 │
                    │  Σ w_k · ΔP_k                │
                    │  (× α = 0.05)                │
                    └──────────┬───────────────────┘
                               │
                               │   ← Added together
                               │
                    ┌──────────┴───────────────────┐
                    │  Clinical Feedback            │
  Patient ────────→ │  E(t) ∈ [-1, +1]             │
                    │  (× β = 0.10)                │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │  f_b(t+Δt) = f_b(t) + sum    │
                    │  clip to [4, 13] Hz          │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │  Binaural Beat Generator     │
                    │  Left:  f₀ + f_b/2           │
                    │  Right: f₀ - f_b/2           │
                    └──────────────────────────────┘
```

**Why two signals?**

1. **EEG signal** ($\alpha$ term): Objective, physiological — measures what the brain is actually doing. Responds to measurable changes in band power.
2. **Clinical signal** ($\beta$ term): Subjective, patient-reported — captures what the patient is feeling. Pain, discomfort, and relief are not always fully reflected in EEG.

By combining both, the controller respects both the physiological reality and the patient experience. The clinical weight ($\beta = 0.10$) is intentionally set higher than the EEG weight ($\alpha = 0.05$) because patient comfort is the primary clinical objective.

---

## 8. Python Implementation (As in the Notebook)

```python
# === Hyperparameters ===
LEARNING_RATES = {
    'alpha': 0.05,   # α: EEG feedback learning rate
    'beta': 0.10,    # β: Clinical feedback learning rate
    'eta': 0.03      # η: Weight update learning rate
}
FREQUENCY_BOUNDS = (4, 13)  # Hz

# === Equation 1: Frequency Update ===
def update_frequency(current_freq, weights, delta_P, clinical_score, learning_rates):
    alpha = learning_rates['alpha']
    beta = learning_rates['beta']

    # Weighted sum of EEG band power changes
    weighted_eeg_change = 0
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        weighted_eeg_change += weights[f'w_{band}'] * delta_P[band]

    # Apply update: f_b(t+Δt) = f_b(t) + α·Σ(w_k·ΔP_k) + β·E(t)
    frequency_change = alpha * weighted_eeg_change + beta * clinical_score
    new_freq = current_freq + frequency_change

    # Clip to [4, 13] Hz
    new_freq = np.clip(new_freq, FREQUENCY_BOUNDS[0], FREQUENCY_BOUNDS[1])
    return new_freq


# === Equation 2: Weight Update ===
def update_weights(weights, current_eeg, target_eeg, learning_rate):
    new_weights = weights.copy()

    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        error = target_eeg[band] - current_eeg[band]
        new_weights[f'w_{band}'] += learning_rate * error

        # Clip weights to [0.1, 2.0]
        new_weights[f'w_{band}'] = np.clip(new_weights[f'w_{band}'], 0.1, 2.0)

    return new_weights


# === Band Power Change Calculation ===
def calculate_eeg_changes(current_eeg, previous_eeg):
    delta_P = {}
    for band in current_eeg.keys():
        delta_P[band] = current_eeg[band] - previous_eeg[band]
    return delta_P
```

---

## 9. Full Control Loop Diagram

```
┌──────────────────────────────────────┐
│  Patient wearing EEG headset         │
│  + stereo headphones                 │
└──────────────┬───────────────────────┘
               │  Raw EEG (62ch @ 250Hz)
               ▼
┌──────────────────────────────────────┐
│  EEG Preprocessing                   │
│  Bandpass 1-40Hz → CAR → Z-Score     │
└──────────────┬───────────────────────┘
               │  Clean EEG
               ▼
┌──────────────────────────────────────┐
│  Band Power Extraction (Welch PSD)   │
│  → δ, θ, α, β, γ proportions        │
│  → Normalized to sum = 1.0           │
└──────────────┬───────────────────────┘
               │  P_k(t) for each band
               ▼
┌──────────────────────────────────────┐
│  Compute ΔP_k = P_k(t) - P_k(t-1)  │
└──────────────┬───────────────────────┘
               │                          ┌─────────────────────┐
               │                          │  Patient Feedback    │
               │                          │  E(t) ∈ [-1, +1]    │
               │                          └──────────┬──────────┘
               ▼                                     │
┌──────────────────────────────────────────────────────┐
│  Adaptive Frequency Controller                       │
│                                                      │
│  f_b(t+Δt) = f_b(t) + α·Σ(w_k·ΔP_k) + β·E(t)     │
│                                                      │
│  clip to [4 Hz, 13 Hz]                              │
└──────────────┬───────────────────────────────────────┘
               │
               ├──────────────────────────────────────┐
               │                                      ▼
               │              ┌────────────────────────────────────┐
               │              │  Online Weight Update              │
               │              │  w_k += η·(P_target - P_observed)  │
               │              │  clip to [0.1, 2.0]               │
               │              └────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Binaural Beat Generator             │
│  Left  ear: f₀ + f_b/2              │
│  Right ear: f₀ − f_b/2              │
└──────────────┬───────────────────────┘
               │  Stereo audio
               ▼
┌──────────────────────────────────────┐
│  Patient Headphones                  │
│  → Frequency-following response      │
│  → Brain entrains to f_b            │
└──────────────┬───────────────────────┘
               │  Changed brain state
               └──── feedback loop closes ──► back to EEG measurement
```

---

## 10. Convergence Behavior

Over the 20-minute session, the system exhibits the following expected behavior:

| Phase              | Minutes | What Happens                                                         |
|--------------------|---------|----------------------------------------------------------------------|
| **Initialization** | 0       | Frequency set based on migraine type + alpha deficit                 |
| **Early**          | 1–3     | Small EEG changes; clinical discomfort ($E < 0$) may pull frequency down |
| **Adaptation**     | 4–10    | Weights converge; alpha weight grows as alpha approaches target      |
| **Convergence**    | 11–15   | Alpha power nears 40% target; frequency stabilizes                   |
| **Plateau**        | 16–20   | System reaches near-equilibrium; small corrections only              |

The weight learning rate $\eta = 0.03$ is intentionally small to ensure gradual, stable adaptation without oscillation.

---

## 11. Design Rationale

### Why Use Band Power Changes ($\Delta P_k$) Instead of Raw Powers?

The controller reacts to **changes** rather than absolute values. This makes it:
- **Responsive to trends:** Rising alpha → positive contribution → frequency reinforces the trend
- **Insensitive to baseline differences:** Different patients have very different absolute band powers. By using deltas, the controller works regardless of the patient's starting state
- **Self-correcting:** If a band overshoots its target, the change becomes negative and the system self-corrects

### Why Online Weight Learning?

Static weights would treat all bands equally throughout the session. Online learning allows the controller to:
- **Focus on what matters:** If alpha is far from target, $w_\alpha$ grows, making frequency updates more responsive to alpha changes
- **De-emphasize achieved goals:** If theta reaches its target, $w_\theta$ stabilizes and no longer drives frequency changes
- **Personalize during the session:** Different patients respond differently to stimulation; weights adapt to each patient's response pattern

### Why Include Clinical Feedback?

EEG alone cannot capture the full patient experience. Pain, comfort, and subjective improvement are clinically important outcomes that may not be fully reflected in band power ratios. The clinical term $\beta \cdot E(t)$ ensures the system respects the patient's experience alongside the objective EEG measurements.

---

## 12. Summary

The adaptive system in `adaptive_migraine_treatment.ipynb` implements a **dual-signal closed-loop controller** with **online weight learning**:

**Frequency Update:**
$$f_b(t+\Delta t) = f_b(t) + \alpha \cdot \left(\sum_{k} w_k(t) \cdot \Delta P_k(t)\right) + \beta \cdot E(t)$$

**Weight Update:**
$$w_k(t+\Delta t) = w_k(t) + \eta \cdot \big(P_{k,\text{target}} - P_{k,\text{observed}}(t)\big)$$

Key properties:
- **Multi-band inputs:** Considers changes across all five EEG frequency bands (δ, θ, α, β, γ)
- **Adaptive weights:** Online learning focuses attention on bands most in need of correction
- **Dual feedback:** Combines objective EEG measurements with subjective patient reports
- **Personalized initialization:** Starting frequency and weights depend on patient demographics and baseline EEG
- **Safe bounds:** Frequency constrained to [4, 13] Hz; weights constrained to [0.1, 2.0]
- **Hyperparameters:** $\alpha = 0.05$, $\beta = 0.10$, $\eta = 0.03$ — all conservative for stable convergence

---

*This document accompanies the FYDP-I project: Personalized Migraine Mitigation via Binaural Beats.*
*Reference implementation: `adaptive_migraine_treatment.ipynb`*
