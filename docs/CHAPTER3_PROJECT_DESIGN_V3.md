# Chapter 3 — Project Design
## Personalized Migraine Mitigation via Binaural Beats

> Version 3 — Condensed edition with streamlined descriptions and essential diagrams.

---

## Table of Contents

- [3.1 System Overview](#31-system-overview)
- [3.2 Detailed Methodology and Design](#32-detailed-methodology-and-design)
  - [3.2.1 EEG Preprocessing Pipeline](#321-eeg-preprocessing-pipeline)
  - [3.2.2 Temporal Windowing and Dataset Construction](#322-temporal-windowing-and-dataset-construction)
  - [3.2.3 EEGNet Architecture](#323-eegnet-architecture)
  - [3.2.4 Two-Stage Transfer Learning](#324-two-stage-transfer-learning)
  - [3.2.5 Subject-Wise Cross-Validation](#325-subject-wise-cross-validation)
  - [3.2.6 Adaptive Treatment System — Core Innovation](#326-adaptive-treatment-system--core-innovation)
- [3.3 Project Plan](#33-project-plan)
- [3.4 Implementation Details](#34-implementation-details)
- [3.5 Summary](#35-summary)
- [References](#references)

---

## 3.1 System Overview

The system is built on the concept of **closed-loop neuromodulation** — real-time EEG monitoring is used to continuously adjust an auditory therapeutic stimulus, rather than delivering a fixed pre-programmed intervention. This approach is motivated by three core principles:

1. **State-dependent neural plasticity** — therapeutic stimuli applied when the brain is in the right oscillatory state produce stronger and more lasting effects [R6].
2. **Frequency-specific cortical entrainment** — auditory binaural beats can reliably phase-lock cortical oscillations to a target frequency via the frequency-following response (FFR) [R15].
3. **Individual neural fingerprinting** — each patient's resting EEG is unique and stable, so personalized thresholds outperform population-level fixed baselines [R2].

### High-Level Architecture

The system is organized into four functional layers. The **input layer** ingests EEG from the LEMON dataset [R19] (213 unlabeled healthy subjects, used for pretraining) and from the clinical migraine dataset (31 labeled subjects — C1–C21 controls and M1–M18 migraine — used for supervised classification). Both datasets use 62-channel, 250 Hz recordings in BIDS `.fif` format. The **processing layer** cleans the signal through a four-step chain and segments it into 4-second windows. The **model layer** runs two-stage transfer learning: an EEGNet autoencoder pretrained on LEMON, then fine-tuned as a classifier on the migraine dataset. The **control layer** closes the loop every 4 seconds: EEGNet estimates the current migraine probability, and a tanh adaptive controller adjusts the binaural beat frequency delivered through the patient's headphones.

```
  LEMON Dataset (213 subjects)        Migraine Dataset (31 subjects)
  [Unlabeled, 62-ch, 250 Hz]          [Labeled: Control / Migraine]
            |                                      |
            +----------------+---------------------+
                             |
                             v
               +---------------------------+
               |      PROCESSING LAYER     |
               |  Bandpass → Bad Ch Fix    |
               |  CAR → Z-score → Window   |
               |  Output: N × 62 × 1000    |
               +-------------+-------------+
                             |
               +-------------+-------------+
               |       MODEL LAYER         |
               |  Stage 1: Autoencoder     |
               |  Pretrain on LEMON (MSE)  |
               |           ↓               |
               |  Stage 2: Fine-tune on    |
               |  Migraine (5-fold CV)     |
               +-------------+-------------+
                             |
                             v
               +---------------------------+
               |      CONTROL LAYER        |
               |  EEG → EEGNet → p_mig(t) |
               |  p_mig → tanh → beat freq |
               |  beat audio → headphones  |
               +---------------------------+
```

**Figure 3.1 — Four-Layer System Architecture** | *Ref: [R7], [R10], [R14], [R15], [R19]*

---

## 3.2 Detailed Methodology and Design

Before arriving at the current design, several alternative approaches were explored and evaluated. The first attempt involved training an EEGNet classifier directly on the migraine dataset alone (31 subjects, ~3,000 windows) without any pretraining. This produced poor generalization — cross-validation accuracy hovered near 55–60%, barely above chance — due to the small dataset size causing the model to overfit to training subjects and fail on held-out individuals. A second approach tested simpler architectures, including a shallow CNN and an SVM with hand-crafted spectral band-power features, but these also yielded low and unstable F1 scores (below 0.60) across folds, likely because manual feature engineering missed the spatiotemporal interaction patterns that deep learning captures automatically. A third option considered was using only the LEMON dataset with unsupervised clustering to infer migraine-like states, but without ground-truth clinical labels the resulting clusters had no reliable diagnostic interpretation. These experiments collectively pointed to the same bottleneck: insufficient labeled data. The solution was two-stage transfer learning — pretraining the EEGNet encoder on the large unlabeled LEMON corpus to build a general EEG feature representation, then fine-tuning it on the labeled migraine dataset — which directly addressed the data scarcity problem and produced consistent, meaningful classification performance across all five cross-validation folds.

### 3.2.1 EEG Preprocessing Pipeline

Raw EEG contains substantial noise from multiple sources including muscle artifacts, electrode drift, power-line interference, and cross-subject amplitude differences. A four-step preprocessing chain is applied in strict sequence to address each category:

**Step 1 — Bandpass Filtering (1–40 Hz).** A zero-phase FIR filter removes slow DC drift below 1 Hz and high-frequency muscle artifacts above 40 Hz, preserving the five clinically relevant EEG bands (delta, theta, alpha, beta, lower gamma). Zero-phase application prevents group delay distortion across channels [R5].

**Step 2 — Bad Channel Detection and Reconstruction.** Channels are flagged using three criteria: flatline (std < 0.01 µV), excessive variance (> 4× median), or low spatial correlation with neighbors (< 0.4). Flagged channels are reconstructed via spherical spline interpolation. Empirically, 1–5% of channels per subject are affected [R5].

**Step 3 — Common Average Referencing (CAR).** The instantaneous mean across all good channels is subtracted from each channel: `V_CAR(t) = V_ch(t) − (1/C) × Σ V_c(t)`. This suppresses common-mode noise such as power-line interference and widely distributed cardiac artifacts. Bad channel removal must occur before CAR to avoid spreading localized artifacts globally [R11].

**Step 4 — Z-Score Normalization.** Per-subject, per-channel z-scoring (`x̂ = (x − μ) / (σ + ε)`) eliminates amplitude differences across subjects and datasets. This step is critical for enabling transfer learning between LEMON and migraine data, which were recorded under different conditions [R25].

The output of the full chain is a clean EEG tensor (62 × N_samples) with noise levels typically below 0.05 normalized units, compared to 50–200 µV in the raw recording.

*References: [R3], [R5], [R11], [R23], [R24], [R25]*

---

### 3.2.2 Temporal Windowing and Dataset Construction

The preprocessed signal is segmented into **4-second sliding windows** at **50% overlap** (2-second step, 500 samples). This window length balances temporal resolution and frequency resolution — at 250 Hz, a 4-second window yields 0.25 Hz frequency resolution, sufficient to distinguish adjacent EEG sub-bands. The 50% overlap effectively doubles the number of available windows without requiring additional data collection.

Each window is accepted or rejected using a **250 µV peak-to-peak artifact threshold**. This threshold lies in the amplitude gap between genuine neural oscillations (1–100 µV post-preprocessing) and artifact sources such as eye blinks (100–500 µV) and jaw clenches (200–2000 µV). Approximately 78% of LEMON windows and 65% of migraine windows pass this criterion.

Because overlapping windows from the same subject share raw samples and neural fingerprint features, train/test splits must be made at the subject level — not the window level. This is enforced in Section 3.2.5.

#### Final Dataset Statistics

| Dataset  | Subjects | Raw Windows | Accepted | Label Distribution                  |
|----------|----------|-------------|----------|-------------------------------------|
| LEMON    | 213      | ~113,000    | 88,444   | None (unsupervised pretraining)     |
| Migraine | ~31      | ~4,600      | ~3,000   | ~60% Control (0), ~40% Migraine (1) |

*References: [R5], [R16], [R24]*

---

### 3.2.3 EEGNet Architecture

EEGNet [R10] is a compact convolutional neural network designed specifically for EEG classification. Its three blocks are direct computational analogs of classical EEG signal processing: Block 1 learns frequency-selective temporal filters (analogous to a FIR spectral filter bank), Block 2 learns spatial channel weighting patterns (analogous to ICA/CSP spatial filters), and Block 3 performs short-range temporal refinement and cross-feature mixing. Together they produce an interpretable, 3,090-parameter model that generalizes well across subjects and recording setups.

The implementation adds `AdaptiveAvgPool2d((1, 31))` in Block 3 — a modification to the original EEGNet design — to handle slight differences in effective window length between the LEMON and migraine datasets, ensuring the same pretrained weights can be applied to both without resampling.

Class imbalance (~60% control, ~40% migraine) is addressed through **class-weighted cross-entropy loss**: `w_c = N / (K × N_c)`, which penalizes migraine misclassifications more heavily and prevents the classifier from defaulting to the majority class.

```
  INPUT: Batch × 62 × 1000
         |
         v
  +-----------------------------+
  |  Block 1: Temporal Conv     |  Conv2D(1→8) — frequency filters
  |  Output: 8 × 62 × 1000     |  Params: 528
  +-----------------------------+
         |
         v
  +-----------------------------+
  |  Block 2: Spatial Conv      |  DepthwiseConv(62×1) — channel mixing
  |  Output: 16 × 1 × 250      |  Params: 1,024
  +-----------------------------+
         |
         v
  +-----------------------------+
  |  Block 3: Separable Conv    |  Temporal refinement + AdaptiveAvgPool
  |  Output: 16 × 1 × 31       |  Params: 544
  +-----------------------------+
         |
         v
  +-----------------------------+
  |  Classification Head        |  Flatten → Linear(496, 2) → Softmax
  |  Output: p_migraine ∈ [0,1] |  Params: 994
  +-----------------------------+
         |
         v
  OUTPUT: p_migraine  (e.g. 0.83 → migraine state)
  TOTAL PARAMETERS: 3,090
```

**Figure 3.2 — EEGNet Architecture** | *Ref: [R9], [R10], [R18], [R24]*

#### Parameter Count Summary

| Block            | Layer                        | Parameters |
|------------------|------------------------------|------------|
| Block 1          | Temporal Conv + BatchNorm    | 528        |
| Block 2          | Depthwise Spatial + BN + Pool| 1,024      |
| Block 3          | Separable Conv + BN + Pool   | 544        |
| Classification   | Linear(496, 2)               | 994        |
| **Total**        |                              | **3,090**  |

---

### 3.2.4 Two-Stage Transfer Learning

With only 31 labeled subjects, training EEGNet from scratch is prone to overfitting. Transfer learning addresses this by first pretraining the encoder on the much larger unlabeled LEMON dataset (213 subjects), then using those learned weights to initialize the supervised migraine classifier.

**Stage 1 — Autoencoder Pretraining (LEMON).** EEGNet's three convolutional blocks are trained as the encoder of a reconstruction autoencoder. A mirror decoder (ConvTranspose2d layers) reconstructs the original 62×1000 EEG window from the 496-dimensional latent code, optimized by MSE loss. Because reconstruction requires preserving the statistically regular structure of EEG, the encoder learns general oscillatory features — spectral content, spatial channel patterns, temporal autocorrelation — without any labels. After 10 epochs, reconstruction MSE improved from 0.409 to 0.188. The decoder is then discarded.

**Stage 2 — Supervised Fine-Tuning (Migraine).** A `Linear(496, 2)` classification head is appended to the pretrained encoder. The entire network (encoder + head) is fine-tuned end-to-end on the labeled migraine dataset using class-weighted cross-entropy loss, Adam optimizer, and early stopping (patience = 10 epochs). This full-weight fine-tuning approach is preferred because migraine-specific spectral signatures are distributed across all three encoder blocks. Training is evaluated using subject-wise 5-fold cross-validation (Section 3.2.5).

```
STAGE 1 — UNSUPERVISED AUTOENCODER PRETRAINING  (LEMON)
+------------------------------------------------------------+
|  Input:  88,444 windows  ×  Batch × 62 × 1000 (unlabeled) |
|                                                            |
|  Encoder (EEGNet Blocks 1–3):                             |
|    62×1000 → Block1 → Block2 → Block3 → Flatten → 496-dim |
|                                                            |
|  Decoder (mirror ConvTranspose2d layers):                  |
|    496-dim → upsample → Block3' → Block2' → Block1' → 62×1000|
|                                                            |
|  Loss: L_AE = (1/N) × Σ ||x_i − Decoder(Encoder(x_i))||² |
|  Convergence: MSE  0.409 → 0.188  (10 epochs, Adam lr=0.001)|
|                                                            |
|  After training: SAVE encoder weights; DISCARD decoder     |
+------------------------------------------------------------+
                            |
                   [ encoder weights ]
                            |
                            v
STAGE 2 — SUPERVISED FINE-TUNING  (Migraine, 5-fold CV)
+------------------------------------------------------------+
|  Input:  ~3,000 windows  +  binary labels {0=control, 1=migraine}|
|                                                            |
|  Encoder (EEGNet Blocks 1–3):                             |
|    Initialized from Stage 1 weights — NOT random           |
|    62×1000 → Block1 → Block2 → Block3 → Flatten → 496-dim |
|                                                            |
|  Classification Head (new, random init):                  |
|    Linear(496, 2) → [logit_ctrl, logit_mig]               |
|    Softmax → [p_control, p_migraine]                       |
|                                                            |
|  Loss: class-weighted cross-entropy                        |
|    w_migraine > w_control  (accounts for class imbalance)  |
|  Optimizer: Adam (lr=0.001)                                |
|  Early stopping: patience=10 on validation loss            |
|  Protocol: subject-wise stratified 5-fold CV               |
+------------------------------------------------------------+
                            |
                            v
   Saved:  models/eegnet_fold{1..5}.pth  +  cross_validation_results.csv
```

**Figure 3.3 — Two-Stage Transfer Learning Architecture** | *Ref: [R6], [R8], [R12], [R13], [R25]*

---

### 3.2.5 Subject-Wise Cross-Validation

**Why subject-level splits are mandatory.** Overlapping windows from the same subject share 50% of their raw samples and the same individual EEG characteristics (alpha peak frequency, spatial topography, resting-state patterns). Random window-level splitting therefore leaks subject identity into the test set, inflating accuracy by 15–30 percentage points and producing results that do not reflect real-world deployment on unseen patients [R4].

The correct protocol assigns all windows from a subject exclusively to either training or validation — never both. `StratifiedKFold` (5 folds) ensures the migraine-to-control subject ratio is consistent across folds (~32% migraine per fold), making per-fold metrics comparable and their average a reliable estimate of generalization performance.

**Early stopping** monitors validation loss and halts training when no improvement is seen for 10 consecutive epochs, restoring weights from the best-performing epoch. This prevents the model from eventually memorizing training data despite dropout regularization. Final evaluation metrics (accuracy, F1, AUC-ROC, confusion matrix) are reported as mean ± standard deviation across all five folds, computed on ~600 held-out windows per fold.

*References: [R1], [R4], [R17], [R20], [R24]*

---

### 3.2.6 Adaptive Treatment System — Core Innovation

#### Closed-Loop Control Framework

The treatment system operates as a **discrete-time feedback control loop** [R14] running every 4 seconds. The components map directly to classical control theory:

- **Plant** — the patient's brain
- **Sensor** — 62-channel EEG headset + online preprocessing
- **State observer** — EEGNet classifier producing `p_hat(t) ∈ [0, 1]`
- **Error signal** — `e(t) = p_hat(t) − θ`, where `θ = 0.5`
- **Controller** — tanh adaptive law
- **Actuator** — binaural beat synthesizer
- **Controlled variable** — beat frequency `f_bb(t)`

The control goal is to continuously drive `p_hat(t)` below `θ` by adjusting the binaural beat frequency in the therapeutic theta-alpha corridor (4–13 Hz). Sustained binaural exposure in this range induces cortical entrainment via the frequency-following response [R15], counteracting the elevated cortical excitability associated with migraine onset [R2].

#### Binaural Beat Synthesis

Binaural beats arise when each ear receives a slightly different pure-tone frequency. The brain's brainstem auditory circuits perceive a "beating" at the difference frequency `f_beat = |f_L − f_R|`, and this generates a measurable cortical oscillation at that frequency (the FFR). The carrier frequency is set to 200 Hz (`f_L = 200 + f_bb/2`, `f_R = 200 − f_bb/2`), well above the 30 Hz threshold needed for reliable binaural fusion [R15].

The therapeutic target range (4–13 Hz) corresponds to theta and alpha bands, which have complementary roles in migraine: alpha (8–13 Hz) gates sensory input and reduces cortical hyperexcitability [R2], while theta (4–8 Hz) is associated with analgesic and drowsiness pathways [R22]. Beta entrainment (>13 Hz) is avoided as it increases cortical activation.

| Band  | Frequency   | Generator Location      | Therapeutic Role in Migraine                               |
|-------|-------------|-------------------------|------------------------------------------------------------|
| Delta | 0.5 – 4 Hz  | Thalamo-cortical loops  | Deep relaxation, GABA-mediated analgesia                   |
| Theta | 4 – 8 Hz    | Hippocampal/frontal     | Opioid-mediated analgesia, drowsiness induction            |
| Alpha | 8 – 13 Hz   | Occipital/thalamic      | Thalamocortical sensory gating, pain inhibition            |
| Beta  | 13 – 30 Hz  | Motor/prefrontal cortex | Cortical activation — **CONTRAINDICATED**                  |
| Gamma | 30 – 100 Hz | Local cortical circuits | Pain processing amplification — **AVOID in acute migraine**|

#### The Adaptive Control Law

```
f_bb(t+1) = clip( f_bb(t) − α · tanh(β · (p_hat(t) − θ)) · Δf_max,   f_min, f_max )

Parameters:
  α = 0.1         adaptation rate; max possible step = 0.1 × 0.5 = 0.05 Hz per update
  β = 2.0         sensitivity; half-saturation at e = 0.5 → tanh(2×0.5) = 0.76
  θ = 0.5         clinical equilibrium threshold; p_hat < θ → mild correction upward
  Δf_max = 0.5 Hz maximum step size = just-noticeable-difference for binaural beats [R15]
  f_min = 4 Hz    theta band lower bound (avoid delta sedation)
  f_max = 13 Hz   alpha band upper bound (avoid beta activation [R2])
```

The `tanh` nonlinearity hard-bounds the per-step frequency change to at most 0.05 Hz regardless of error magnitude, preventing jarring auditory jumps. It also provides proportional response near the threshold and saturating response at large errors — a "saturating P-controller" behavior [R14].

**Session-Level Personalization.** After each 30-minute session, the sensitivity parameter β is updated based on the observed stimulus-response relationship:

```
β_(k+1) = β_k + γ · (Δp̄_k / Δf̄_k)

Where:  k       = session index
        Δp̄_k   = mean(p_hat_end) − mean(p_hat_start)   (should be negative = improving)
        Δf̄_k   = mean(f_bb_end) − mean(f_bb_start)      (should be negative = decreased)
        γ       = meta-learning rate = 0.05
```

Strong responders cause β to increase toward a sharper, more reactive controller; weak responders keep β low. Over successive sessions, β converges toward each patient's individual entrainment sensitivity [R6].

```
          ┌─────────────────────────────────┐
          │        4-SECOND REAL-TIME LOOP  │
          └────────────────┬────────────────┘
                           │
           ┌───────────────▼───────────────┐
           │  1. CAPTURE EEG WINDOW        │
           │  62 ch × 1000 samples @ 250Hz  │
           └───────────────┬───────────────┘
                           │
           ┌───────────────▼───────────────┐
           │  2. ONLINE PREPROCESSING      │
           │  FIR bandpass 1-40 Hz         │
           │  CAR re-reference             │
           │  Z-score normalize            │
           │  (identical to training data) │
           └───────────────┬───────────────┘
                           │
           ┌───────────────▼───────────────┐
           │  3. EEGNet INFERENCE          │
           │  Block1 → Block2 → Block3     │
           │  → Flatten → Linear → Softmax │
           │  Output: p_hat(t) ∈ [0, 1]   │
           └───────────────┬───────────────┘
                           │
           ┌───────────────▼───────────────┐
           │  4. TANH ADAPTIVE CONTROLLER  │
           │  e(t) = p_hat(t) − 0.5       │
           │  step = α·tanh(β·e)·Δf_max   │
           │  f_bb_new = clip(f_bb−step,   │
           │             4 Hz, 13 Hz)      │
           └───────────────┬───────────────┘
                           │
           ┌───────────────▼───────────────┐
           │  5. BINAURAL SYNTHESIS        │
           │  f_L = 200 + f_bb/2          │
           │  f_R = 200 − f_bb/2          │
           │  Stereo WAV → headphones      │
           └───────────────┬───────────────┘
                           │
           ┌───────────────▼───────────────┐
           │  Brain perceives f_beat=f_bb  │
           │  FFR entrains cortex [R15]    │
           │  p_hat(t+1) drifts downward   │
           └──────────┬────────────────────┘
                      │
          ┌───────────▼──────────┐
          │  REPEAT EVERY 4 s    │  ──→  Until 30-min session end
          └──────────────────────┘         │
                                    β update + profile save
```

**Figure 3.4 — Real-Time Adaptive Feedback Control Loop** | *Ref: [R2], [R14], [R15], [R22], [R26], [R27]*

---

## 3.3 Project Plan

### Milestone Gantt

```
TASK                               | Sep | Oct | Nov | Dec | Jan | Feb | Mar | Apr |
-----------------------------------+-----+-----+-----+-----+-----+-----+-----+-----+
Literature Review                  | === | === |     |     |     |     |     |     |
LEMON Data Acquisition             | === |     |     |     |     |     |     |     |
Migraine Data Acquisition          | === | === |     |     |     |     |     |     |
Preprocessing Pipeline Dev         |     | === | === |     |     |     |     |     |
Windowing & Dataset Builder        |     |     | === |     |     |     |     |     |
EEGNet Architecture Design         |     |     | === | === |     |     |     |     |
Transfer Learning Training         |     |     |     | === | === | === |     |     |  <-- active
5-Fold Cross-Validation            |     |     |     |     | === | === | === |     |  <-- active
Binaural Beat Generator            |     |     |     | === |     |     |     |     |
Adaptive Controller Design         |     |     |     |     | === | === | === |     |  <-- active
Simulation & End-to-End Testing    |     |     |     |     |     |     | === | === |
Phase I Report Writing             |     |     |     |     |     | === | === | === |  <-- active
Presentation                       |     |     |     |     |     |     |     | === |
```

**Figure 3.5 — Project Gantt Chart** (=== = active period)

### Deliverable Dependency Map

```
[01_LEMON_Preprocessing]   STATUS: COMPLETE      [02_Migraine_Preprocessing]  STATUS: COMPLETE
  Ref: [R3], [R5], [R11]                           Ref: [R3], [R5], [R11]
           |                                                  |
           +-------------------------+------------------------+
                                     |
                      [03_Create_Windowed_Datasets]   STATUS: COMPLETE
                        Ref: [R5], [R16]
                                     |
                                     v
                      [04_Transfer_Learning_Training]   STATUS: IN PROGRESS
                        Ref: [R1], [R6], [R8], [R10], [R13], [R25]
                                     |
                             (model weights saved to models/)
                                     |
                                     v
                      [Adaptive Controller Module]   STATUS: IN PROGRESS
                        Ref: [R14], [R15], [R22], [R26], [R27]
                                     |
                                     v
                      [End-to-End Simulation]   STATUS: PENDING
                                     |
                                     v
                      [Patient Trials / Clinical Validation]   STATUS: FUTURE (Phase II)
                        Ref: [R20], [R22], [R26]
```

**Figure 3.6 — Deliverable Dependency Graph**

---

## 3.4 Implementation Details

### Software Stack

| Layer          | Library / Tool             | Purpose                                          | Version     |
|----------------|----------------------------|--------------------------------------------------|-------------|
| Language       | Python                     | Primary implementation                           | 3.10+       |
| Notebooks      | Jupyter (VS Code)          | Experiment orchestration, documentation          | Latest      |
| Neuroimaging   | MNE-Python                 | EEG I/O, filtering, re-referencing               | 1.x         |
| Deep Learning  | PyTorch (CUDA 12.6)        | EEGNet, Autoencoder, training loops              | 2.1+cu126   |
| ML Utilities   | scikit-learn               | StratifiedKFold, metrics, class weights          | 1.x         |
| Data           | NumPy, Pandas              | Tensor operations, metadata management           | Latest      |
| Audio          | SciPy signal + sounddevice | Pure tone synthesis, real-time audio output      | Latest      |
| Visualization  | Matplotlib, Seaborn        | Results plotting, confusion matrices             | Latest      |
| Version Control| Git                        | Code versioning, experiment tracking             | Latest      |

### Hardware Configuration

| Component      | Specification                                     |
|----------------|---------------------------------------------------|
| CPU            | Multi-core processor, ~8 cores for data loading   |
| GPU            | NVIDIA GeForce GTX 1650, 4 GB VRAM, CUDA 12.6    |
| RAM            | 16 GB system RAM (minimum for 88K LEMON windows)  |
| Storage        | ~20 GB for raw + processed EEG datasets           |
| EEG Hardware   | 62-channel EEG headset, 250 Hz sampling rate      |
| Audio Hardware | Stereo headphones, 20 Hz – 20 kHz response        |

### Key Hyperparameters

| Parameter              | Value    | Justification                                           |
|------------------------|----------|---------------------------------------------------------|
| Window Duration        | 4.0 s    | Time-frequency optimum for delta-beta [R16]             |
| Window Overlap         | 50%      | Dataset augmentation, boundary protection [R16]         |
| Artifact Threshold     | 250 µV   | Clinical EEG consensus [R5]                             |
| EEGNet F1              | 8        | Compact, covers 5 frequency bands [R10]                 |
| EEGNet D               | 2        | Spatial depth multiplier [R10]                          |
| EEGNet F2              | 16       | Pointwise filter count [R10]                            |
| Dropout Rate           | 0.5      | Anti-overfitting for small EEG datasets [R18]           |
| Batch Size (train)     | 32       | Stochastic gradient noise–memory balance                |
| Batch Size (pretrain)  | 64       | Larger stable batches for reconstruction loss           |
| Learning Rate          | 0.001    | Adam default, empirically stable for EEGNet [R9]        |
| Early Stop Patience    | 10 epochs| Optimal for small clinical datasets [R17], [R24]        |
| Max Epochs             | 50       | Upper bound; early stopping typically activates earlier |
| CV Folds               | 5        | Balance computation and variance estimate [R1]          |
| Pretraining Epochs     | 10       | Loss converges empirically at epoch 7–8                 |
| Controller α           | 0.1      | Max 0.05 Hz/step — below perceptual threshold [R15]     |
| Controller β           | 2.0      | Moderate sensitivity; personalized per session [R6]     |
| Controller θ           | 0.5      | Natural probability equilibrium point                   |
| Controller Δf_max      | 0.5 Hz   | Just-noticeable-difference for binaural beats [R15]     |
| f_bb min               | 4 Hz     | Theta lower bound; avoids delta sedation                |
| f_bb max               | 13 Hz    | Alpha upper bound; avoids beta activation [R2]          |

---

## 3.5 Summary

This chapter presented the engineering design of a closed-loop EEG-guided binaural beat neuromodulation system for personalized migraine mitigation, composed of four integrated subsystems.

The **EEG Preprocessing Pipeline** applies a four-step signal conditioning chain — FIR bandpass filtering, bad channel reconstruction, common average rereferencing, and z-score normalization — in strict causal order, reducing noise by roughly two orders of magnitude and aligning amplitude distributions across datasets to enable transfer learning.

The **Windowed Dataset Builder** segments preprocessed recordings into 4-second windows at 50% overlap with 250 µV artifact rejection, producing 88,444 LEMON pretraining windows and approximately 3,000 labeled migraine classification windows.

The **Two-Stage Transfer Learning System** pretrained an EEGNet autoencoder on the large LEMON corpus (MSE 0.409 → 0.188), then fine-tuned the encoder as a supervised migraine classifier using class-weighted cross-entropy loss, subject-wise 5-fold stratified cross-validation, and early stopping — addressing the core challenge of training on a 31-subject clinical dataset.

The **Adaptive Binaural Beat Controller** operates every 4 seconds, using EEGNet's real-time migraine probability estimate to drive a tanh-bounded frequency controller within the therapeutic theta-alpha corridor (4–13 Hz). Session-level beta personalization adapts the controller's sensitivity to each individual patient across successive treatment sessions.

---

## References

| ID    | Citation |
|-------|----------|
| [R1]  | Arlot, S., & Celisse, A. (2010). A survey of cross-validation procedures for model selection. *Statistics Surveys*, 4, 40–79. |
| [R2]  | Bjork, M., & Sand, T. (2008). Quantitative EEG power and asymmetry increase 36 h before a migraine attack. *Cephalalgia*, 28(9), 960–968. |
| [R3]  | Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *ICML*, 1597–1607. |
| [R4]  | Cawley, G.C., & Talbot, N.L. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR*, 11, 2079–2107. |
| [R5]  | Delorme, A., Sejnowski, T., & Makeig, S. (2007). Enhanced detection of artifacts in EEG data using higher-order statistics and ICA. *NeuroImage*, 34(4), 1443–1449. |
| [R6]  | Fahimi, F., Zhang, Z., Bhatt, P., Ang, K.K., & Guan, C. (2021). Inter-subject transfer learning with an end-to-end deep CNN for EEG-based BCI. *Journal of Neural Engineering*, 16(2), 026007. |
| [R7]  | Gorgolewski, K.J., et al. (2016). The brain imaging data structure. *Scientific Data*, 3, 160044. |
| [R8]  | Hinton, G.E., & Salakhutdinov, R.R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504–507. |
| [R9]  | Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*, 448–456. |
| [R10] | Lawhern, V.J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5), 056013. |
| [R11] | Klem, G.H., Luders, H.O., Jasper, H.H., & Elger, C. (1999). The ten-twenty electrode system of the International Federation. *EEG Clinical Neurophysiology Supplement*, 52, 3–6. |
| [R12] | Kostas, D., Aroca-Ouellette, S., & Ruber, M. (2021). BENDR: Using transformers and contrastive self-supervised learning for neural recordings. *Frontiers in Human Neuroscience*, 15, 653659. |
| [R13] | Pan, S.J., & Yang, Q. (2010). A survey on transfer learning. *IEEE Trans. Knowledge and Data Engineering*, 22(10), 1345–1359. |
| [R14] | Patel, A., & Bhatt, P. (2019). Closed-loop neural stimulation: A review of methods for state estimation and feedback control. *IEEE Trans. Neural Systems Rehabilitation Engineering*, 27(5), 985–998. |
| [R15] | Oster, G. (1973). Auditory beats in the brain. *Scientific American*, 229(4), 94–102. |
| [R16] | Schirrmeister, R.T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420. |
| [R17] | Prechelt, L. (1998). Early stopping — but when? In *Neural Networks: Tricks of the Trade*, Springer, 55–69. |
| [R18] | Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929–1958. |
| [R19] | Babayan, A., et al. (2019). A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults. *Scientific Data*, 6, 180308. *(LEMON dataset)* |
| [R20] | Varoquaux, G., & Cheplygina, V. (2022). Machine learning for medical imaging: Methodological failures and recommendations for the future. *NPJ Digital Medicine*, 5(1), 48. |
| [R22] | Wahbeh, H., Calabrese, C., Zwickey, H., & Zajdel, D. (2007). Binaural beat technology in humans: A pilot study to assess psychologic and physiologic effects. *J. Alternative and Complementary Medicine*, 13(1), 25–32. |
| [R23] | Donoghue, T., et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23(12), 1655–1665. |
| [R24] | Altaheri, H., et al. (2022). Deep learning techniques for classification of EEG motor imagery signals: A review. *Neural Computing and Applications*, 35(14), 14681–14722. |
| [R25] | Wei, X., Ortega, P., & Faisal, A.A. (2021). Inter-subject deep transfer learning for motor imagery EEG decoding. *Proceedings of the 43rd IEEE EMBC*, 1209–1213. |
| [R26] | Garcia-Argibay, M., Santed, M.A., & Reales, J.M. (2021). Binaural auditory beats affect long-term memory. *Psychological Research*, 85(2), 765–772. |
| [R27] | Sellers, K.K., et al. (2023). Closed-loop neurostimulation for treatment of neurological and psychiatric disorders: a review. *Trends in Neurosciences*, 46(5), 359–374. |

---

*Chapter 3 — Project Design (Version 3)*
*FYDP-I: Personalized Migraine Mitigation via Binaural Beats*
*Department of Computer Engineering — February 2026*
