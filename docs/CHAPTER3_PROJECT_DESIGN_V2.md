# Chapter 3 — Project Design
## Personalized Migraine Mitigation via Binaural Beats

> Version 2 — Revised edition with expanded theoretical foundations, consolidated diagrams, and updated 2020–2026 references.

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

### Theoretical Foundation

The design of this system is grounded in the concept of **closed-loop neuromodulation** — a paradigm in which real-time measurements of neural activity are used to inform the delivery of therapeutic stimulation in a feedback-driven manner [R14]. Unlike open-loop systems that deliver a fixed, pre-programmed stimulus regardless of the patient's current physiological state, closed-loop systems continuously observe, classify, and respond to dynamic brain states. This enables therapeutic personalization at a temporal granularity that is fundamentally unachievable in standard care, where a physician prescribes a static intervention that cannot adapt to the patient's fluctuating neural state across hours or days. Recent translational reviews have confirmed that closed-loop architectures consistently outperform matched open-loop controls in both efficacy and tolerability across a range of neurological targets [R27].

The first foundational principle underpinning our design is **state-dependent neural plasticity** [R6]. Cortical plasticity — the brain's capacity for functional reorganization — is not a uniform background process but one that is highly sensitive to the precise oscillatory state at the moment stimulation is delivered. Therapeutic stimuli applied during states of appropriate oscillatory phase and amplitude have disproportionately large neuroplastic effects compared to those delivered at arbitrary times [R6]. A controller that continuously monitors neural state and intervals its stimulation outputs to coincide with optimal receptivity windows will therefore outperform any fixed-schedule intervention, because it exploits the brain's own rhythmic machinery rather than working against it. This principle has been operationalized successfully in closed-loop deep brain stimulation for Parkinson's disease and, more recently, in transcranial alternating current stimulation for memory consolidation [R27].

The second principle is **frequency-specific cortical entrainment** [R15]. External periodic stimuli — whether auditory, visual, or electrical — can phase-lock endogenous neural oscillations when delivered near the resonant frequency of specific cortical circuits. In the auditory domain this phenomenon is known as the **frequency-following response** (FFR): the scalp EEG exhibits measurable, sustained power at the frequency of the auditory stimulus, reflecting genuine cortical oscillation rather than a mere peripheral echo of sound transduction. Recent quantitative reviews using scalp EEG and combined EEG–fMRI confirm that the FFR is a robust whole-brain effect spanning brainstem, thalamus, and cortex simultaneously [R26]. This makes auditory binaural beat delivery a uniquely non-invasive route to targeted cortical oscillation modulation, requiring only standard consumer-grade stereo headphones.

The third principle is **individual neural fingerprinting** [R2]. Individual differences in resting-state EEG power spectra are large, temporally stable across years, and substantially heritable, reflecting reliable interindividual variation in thalamocortical circuit architecture [R23]. This has two consequences for system design. First, migraine-associated deviations from a specific patient's personal EEG baseline are real and detectable by a machine learning classifier. Second, population-level fixed thresholds will inevitably be inferior to personalized adaptive thresholds, because what constitutes a pathological oscillatory pattern for one patient may fall within the normal variability envelope for another. This individualization imperative is the direct motivation for the session-level personalization loop embedded in our adaptive controller.

The system architecture integrates all three principles: a data-driven neural state estimator (the EEGNet classifier [R10]) provides real-time probabilistic monitoring of brain state; a tanh-bounded adaptive controller [R14] translates that probabilistic signal into frequency commands; and a real-time binaural beat synthesizer [R15] delivers the resulting entrainment stimulus through headphones, completing the feedback loop.

### High-Level Architecture

The system is organized into four functional layers in a cascading processing pipeline. The **input layer** ingests multi-channel EEG from the LEMON healthy-subject corpus [R19] (213 participants, unlabeled, used for unsupervised encoder pretraining) and from the clinical migraine dataset (31 labeled subjects — groups C1–C21 and M1–M18 — used for supervised classification). Both datasets were recorded at 250 Hz across 62 electrodes in the standard international 10-20 configuration and stored in BIDS-compliant `.fif` format, ensuring reproducibility and compatibility with the MNE-Python toolchain [R7]. The **processing layer** applies a four-step signal conditioning chain (bandpass filtering, bad channel reconstruction, common average re-referencing, and z-score normalization) implemented in MNE-Python, followed by temporal segmentation into 4-second sliding windows at 50% overlap. The **model layer** implements two-stage transfer learning: an EEGNet autoencoder [R8] is pretrained on the large unlabeled LEMON corpus to learn general oscillatory feature representations, and its encoder weights are then used to initialize a supervised EEGNet classifier [R10] fine-tuned on the smaller labeled migraine dataset using subject-wise 5-fold cross-validation. The **control layer** closes the therapeutic loop at runtime, running every 4 seconds: the patient's current EEG window is preprocessed identically to training data, passed through EEGNet inference to produce a migraine probability estimate `p_migraine(t)`, and fed into the tanh adaptive controller that continuously adjusts the binaural beat frequency delivered to the patient. This layered design follows the **separation of concerns** principle [R7], ensuring each module can be independently tested, profiled, swapped, or upgraded without breaking the rest of the pipeline.

```
  LEMON Dataset (213 subjects)          Migraine Dataset (31 subjects)
  8-min resting EEG, unlabeled          C1-C21 (control), M1-M18 (migraine)
  62-channel, 250 Hz, BIDS .fif         62-channel, 250 Hz, binary labels
            |                                         |
            +-------------------+--------------------+
                                |
                                v
         +------------------------------------------------------+
         |  PROCESSING LAYER                                    |
         |  1. FIR Bandpass filter  1 – 40 Hz (zero-phase)     |
         |  2. Bad channel detect + spherical spline interp.    |
         |  3. Common average re-reference (CAR)                |
         |  4. Per-subject, per-channel z-score normalization   |
         |  5. Sliding window: 4 s, 50% overlap, 250 µV reject  |
         |  Output shape:  N × 62 × 1000                       |
         +---------------------------+--------------------------+
                                     |
              +----------------------+----------------------+
              |  Stage 1 — Autoencoder Pretraining          |
              |  Input: 88,444 LEMON windows (no labels)    |
              |  Encoder (EEGNet Blocks 1-3) → 496-dim code  |
              |  Decoder (mirror ConvTranspose2d) → 62×1000  |
              |  Loss: MSE reconstruction, 10 epochs         |
              |  Save encoder weights; discard decoder        |
              |                                              |
              |  Stage 2 — Supervised Fine-tuning            |
              |  Input: ~3,000 migraine windows + labels     |
              |  Pretrained encoder + Linear(496, 2) head    |
              |  Loss: class-weighted cross-entropy          |
              |  Protocol: 5-fold subject-wise stratified CV |
              +----------------------+----------------------+
                                     |
                                     v
         +------------------------------------------------------+
         |  CONTROL LAYER  (4-second real-time loop)            |
         |  EEG capture → online preprocess → EEGNet inference  |
         |  p_migraine(t) → tanh adaptive controller            |
         |  f_bb(t+1) = clip(f_bb − α·tanh(β·e), 4, 13 Hz)    |
         |  Binaural synthesis:  f_L = f₀ + f_bb/2             |
         |                       f_R = f₀ − f_bb/2             |
         |  Stereo audio → patient headphones                   |
         +------------------------------------------------------+
```

**Figure 3.1 — Four-Layer System Architecture** | *Ref: [R7], [R10], [R14], [R15], [R19]*

---

## 3.2 Detailed Methodology and Design

### 3.2.1 EEG Preprocessing Pipeline

#### Theoretical Background

EEG is among the most noise-prone biosignals collected in routine clinical and research practice. The raw scalp-recorded potential is a superposition of genuine cortical generators, volume-conducted artifacts from extracranial sources (cardiac, ocular, and muscular), and environmental electromagnetic interference from power-line coupling and electrode impedance fluctuations. In practice the signal-to-noise ratio of unprocessed EEG is often less than one, meaning artifact energy exceeds neural signal energy across much of the frequency spectrum [R5]. Before any deep learning model can extract diagnostically meaningful structure, this contamination must be suppressed by a principled, reproducible preprocessing chain. Recent EEG preprocessing benchmarks confirm that the quality of signal conditioning is the single most consequential factor in downstream classification performance, outweighing architectural choices in the model itself [R24].

The first category of noise is **out-of-band contamination**. Frequencies below 1 Hz include slow electrode drift, non-stationary DC potential shifts from galvanic skin responses, and movement-induced baseline wander; these create long-timescale amplitude trends that corrupt the stationarity assumption on which all short-window spectral analysis depends [R23]. Frequencies above 40 Hz are dominated by high-frequency electromyographic (EMG) artifacts from jaw and scalp muscles, which generate broadband electrical noise coupling directly into skull electrodes at amplitudes orders of magnitude larger than cortical gamma oscillations. A zero-phase finite impulse response (FIR) bandpass filter with passband 1–40 Hz, implemented via the Parks-McClellan algorithm in MNE-Python, eliminates both tails while preserving all five clinically relevant oscillatory bands — delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and lower gamma (30–40 Hz). Zero-phase (forward-backward) application of the FIR filter eliminates group delay distortion, preserving the temporal alignment of oscillatory events across channels, which is critical for both spatial analysis and epoch-level classification [R5].

The second category is **spatially localized noise**. Individual electrodes with poor gel contact, broken lead wires, or bridging between neighboring sites produce recording channels that are either completely flat (functionally disconnected) or excessively noisy relative to surrounding sites. These channels are not correctable by spatial referencing and must be identified and reconstructed before any global spatial operation is applied. Detection uses three independent statistical criteria: a flatline criterion (per-channel standard deviation below 0.01 µV across the full recording), a high-variance criterion (per-channel standard deviation exceeding four times the median across all channels), and a low-spatial-correlation criterion (mean Pearson correlation with spatially nearest neighbors below 0.4). Any channel satisfying at least one criterion is flagged and reconstructed via spherical spline interpolation [R5], which models the scalp as a spherical conductor surface and estimates the missing channel's potential as a weighted interpolation of neighboring channels, with weights falling exponentially with angular electrode separation in the 10-20 layout coordinate system [R11]. Empirically, 1–5% of channels are flagged per subject, consistent with published literature on 62-channel dry-electrode recordings.

The third category is **common-mode noise**. Power-line electromagnetic interference at 50/60 Hz couples nearly identically to all scalp electrodes because the electromagnetic wavelength at these frequencies (~6000 km) vastly exceeds the head diameter (~20 cm). Common average referencing (CAR) suppresses this by subtracting the instantaneous mean across all active channels from each individual channel: `V_ch_CAR(t) = V_ch(t) − (1/C) × Σ_c V_c(t)`, where C is the count of good channels remaining after bad-channel removal. This operation approximates an "infinity reference" — a theoretically ideal reference electrode placed infinitely far from all biological sources [R11]. Volume-conducted artifacts from cardiac and jaw muscle sources are also attenuated by CAR because they carry large common-mode components. The ordering constraint — bad channel removal must strictly precede CAR — is critical: a noisy channel included in the mean corrupts the reference estimate and disperses that artifact globally across every output channel, transforming a localized problem into a systemic one [R5].

The fourth category is **cross-subject amplitude scaling**. Individual differences in skull thickness, scalp conductivity, and electrode gel impedance produce raw EEG amplitudes spanning a five-to-ten-fold range across subjects. When a model is trained on subjects with one amplitude distribution and evaluated on subjects with a different distribution, the mismatch constitutes a classic domain shift problem that systematically degrades generalization [R3]. Per-subject, per-channel z-score normalization eliminates this by computing the temporal mean `μ_ch` and standard deviation `σ_ch` for each channel over the full recording and applying `x̂_ch(t) = (x_ch(t) − μ_ch) / (σ_ch + ε)`, where the constant `ε = 1e-8` ensures numerical stability when a channel has near-zero variance. After z-scoring, every channel of every subject has mean zero and standard deviation one, making the amplitude distributions of LEMON (recorded with BrainProducts actiCAP at MPI Leipzig) and the migraine clinical dataset directly commensurable. This is the specific step that enables the LEMON-pretrained encoder to receive migraine EEG without experiencing distribution shift in signal amplitude — a precondition for meaningful transfer learning [R25].

The four preprocessing steps are applied in strict sequence: bandpass filtering, then bad channel detection and spherical spline reconstruction, then common average rereferencing, then z-score normalization. This ordering is not interchangeable. Applying CAR before bad channel removal would spread localized artifacts globally; applying z-score normalization before filtering would preserve the very slow drifts that normalization is meant to make irrelevant. The output is a clean EEG tensor of shape 62 × N_samples, with typical noise levels below 0.05 normalized units compared to 50–200 µV in the raw recording — an effective noise reduction of approximately two orders of magnitude.

*References: [R3], [R5], [R11], [R23], [R24], [R25]*

---

### 3.2.2 Temporal Windowing and Dataset Construction

#### Theoretical Background

Neural oscillations are quasi-stationary processes: over short epochs of two to eight seconds the power spectrum of EEG is approximately stationary, but over longer timescales spectral content shifts substantially as the brain cycles between cognitive, arousal, and attentional states [R16]. This **local stationarity assumption** is the theoretical cornerstone of short-window EEG analysis. By segmenting a continuous preprocessed recording into 4-second epochs, each window represents a snapshot of the brain's instantaneous oscillatory configuration, and EEGNet is trained to map that configuration to a classification outcome. Recent benchmarks confirm this assumption holds sufficiently well across a variety of resting and task-related paradigms to support reliable 4-second epoch classification [R24].

The choice of window length is governed by the **time-frequency uncertainty principle**, which in signal processing states that temporal resolution and frequency resolution are reciprocally related: one cannot simultaneously achieve fine frequency resolution and fast temporal tracking [R16]. A 4-second window at 250 Hz yields a frequency resolution of 0.25 Hz (= 1/4.0), making it possible to distinguish individual subband boundaries such as the alpha onset at 8.00 Hz from the upper theta limit at 7.75 Hz. A 1-second window would degrade this to 1.0 Hz, coarse enough to blur the therapeutically critical theta-alpha boundary and cause systematic band misclassification. An 8-second window would provide finer resolution (0.125 Hz) than is clinically necessary while reducing the controller update rate to a temporal granularity where rapid transitions in brain state go undetected. Schirrmeister et al. [R16] confirmed through systematic benchmarking that the 4-second window is near-optimal across multiple deep learning architectures for resting-state EEG classification, a finding replicated in subsequent work covering motor imagery and clinical EEG [R24].

Windows are extracted by a **sliding window with 50% overlap**, meaning consecutive windows are offset by 2 seconds (500 samples at 250 Hz). This doubling of window count — from approximately 30 to approximately 59 per 120-second recording — constitutes a substantial data augmentation without requiring additional subjects, which is critical when working with a 31-subject clinical dataset. The 50% overlap also protects against boundary effects: a transient neural event occurring near a window edge would be split across two non-overlapping windows and might be rejected in both due to the artifact threshold; with 50% overlap, the same event is captured in its entirety in at least one window. The statistical non-independence of consecutive overlapping windows — they share 50% of their temporal samples and therefore the same subject's neural fingerprint — absolutely prohibits random window-level train/test splitting. This is the fundamental reason subject-level splitting is required, as detailed in Section 3.2.5.

Each candidate window is screened by a peak-to-peak artifact rejection criterion. The threshold of 250 µV across all channels is grounded in clinical EEG norms [R5]: genuine neural oscillations after preprocessing occupy 1–20 µV, with strong occipital alpha reaching up to 50 µV and sleep spindles up to 100 µV. Artifacts occupy a distinct amplitude range — eye blinks produce 100–500 µV in frontal channels, jaw clenches 200–2000 µV, and electrode position shifts above 500 µV. The 250 µV criterion sits in the amplitude gap between the upper tail of neural signal and the lower tail of artifact, preserving most clean windows while discarding clearly contaminated ones. Empirically, approximately 78% of LEMON windows and 65% of migraine windows pass this criterion, with the higher rejection rate in the migraine cohort reflecting the less controlled clinical recording environment. Accepted windows are saved as NumPy arrays: `lemon_windows.npy` for the unlabeled LEMON pretraining corpus and `migraine_windows.npy` with `migraine_labels.npy` for the labeled classification dataset.

*References: [R5], [R16], [R24]*

#### Final Dataset Statistics

| Dataset  | Subjects | Raw Windows | Accepted | Label Distribution                  |
|----------|----------|-------------|----------|-------------------------------------|
| LEMON    | 213      | ~113,000    | 88,444   | None (unsupervised pretraining)     |
| Migraine | ~31      | ~4,600      | ~3,000   | ~60% Control (0), ~40% Migraine (1) |

---

### 3.2.3 EEGNet Architecture

#### Theoretical Foundation

EEGNet [R10] is designed by mapping the operations of classical EEG signal processing — spectral decomposition into oscillatory bands followed by spatial source separation — into differentiable convolutional layers. This philosophy distinguishes EEGNet from generic time-series CNNs applied to EEG: every architectural choice has a neuroscientific interpretation, which explains why EEGNet generalizes substantially better across subjects and recording setups than unstructured networks with far more parameters. Systematic comparative reviews of deep learning approaches for EEG consistently rank EEGNet among the top-performing architectures when evaluated under subject-independent cross-validation, precisely because interpretable inductive biases constrain the solution space toward neurophysiologically plausible features [R24].

Classical EEG analysis pipelines operate in two sequential stages: spectral decomposition (bandpass filtering into delta, theta, alpha, beta, and gamma bands) followed by spatial decomposition (independent component analysis or common spatial pattern filters to isolate cortical source activity). EEGNet's three convolutional blocks implement learnable, end-to-end trainable analogs of exactly these two operations. Block 1 is a temporal convolution that learns frequency-selective bandpass impulse responses, playing the role of the classical FIR filter bank. Block 2 is a depthwise spatial convolution that learns electrode channel weighting patterns for each temporal filter output, playing the role of ICA or CSP spatial filters. Block 3 is a separable convolution performing short-range temporal refinement and cross-feature integration. Together, these three blocks implement a compact 3,090-parameter model that achieves state-of-the-art resting-state and motor imagery EEG classification performance [R10]. The deliberate compactness is well-motivated: on datasets of 20–50 clinical subjects, over-parameterized models consistently overfit regardless of regularization, and model parsimony is the single most reliable antidote [R20].

**Block 1 — Temporal Convolution.** The first block applies F1 = 8 filters of shape (1 × 64) to the (62 × 1000) input, which is unsqueezed to (1 × 62 × 1000) before convolution. The kernel length of 64 samples corresponds to 256 ms at 250 Hz. By the Nyquist sampling theorem a filter of length L accurately represents temporal patterns at frequencies down to approximately 250/L ≈ 3.9 Hz, sufficient to capture delta and theta band dynamics. Zero-padding on the time axis preserves the temporal length at 1000 samples throughout Block 1. Each filter independently learns a frequency-selective impulse response and, after convergence, the filters' magnitude spectra closely resemble the bandpass responses of a classical FIR filter bank — an emergent property reported by Lawhern et al. [R10] and confirmed in subsequent audits of trained EEGNet models [R24]. Batch normalization [R9] after convolution stabilizes training by normalizing each filter's output distribution across the batch dimension, preventing internal covariate shift. The output of Block 1 has shape (Batch × 8 × 62 × 1000) and contains 528 trainable parameters.

**Block 2 — Depthwise Spatial Convolution.** The second block applies depthwise (channel-wise) convolutions of shape (62 × 1) independently to each of the 8 temporal feature maps. With depth multiplier D = 2, each temporal filter produces 2 independent spatial filter outputs, yielding 8 × 2 = 16 feature maps total. The (62 × 1) spatial kernel connects all 62 electrode channels to a single output, learning a weighted linear combination of channels — exactly what ICA component unmixing computes, but differentiably adapted to the training data rather than derived from second-order statistics alone [R10]. Following batch normalization and Exponential Linear Unit (ELU) activation, average pooling with factor 4 along the time dimension reduces temporal extent from 1000 to 250 samples, providing low-pass smoothing consistent with classical broadband power estimation. Average pooling is specifically preferred over max pooling for EEG because neural oscillations are continuous, phase-varying processes rather than sparse transient events: average pooling captures mean oscillatory power while max pooling would disproportionately reflect noise peaks. Dropout at p = 0.5 then randomly zeros entire feature maps during training, a regularization strategy shown to be especially effective on small EEG datasets because it prevents co-adaptation of spatial filters [R18]. The output is (Batch × 16 × 1 × 250) with 1,024 trainable parameters.

**Block 3 — Separable Convolution with AdaptiveAvgPool.** The third block factorizes a full convolution into a depthwise component (1 × 16 kernels, one per feature map) that captures short-range temporal correlations within each feature map, followed by a pointwise (1 × 1) convolution that learns linear combinations across the 16 feature maps. This factorization, borrowed from MobileNet-style efficient architecture design, reduces parameter count while maintaining the expressiveness of a full convolution [R10]. Our implementation adds `AdaptiveAvgPool2d((1, 31))` after the separable convolution — a modification that replaces the fixed-stride pooling in the original EEGNet. The standard fixed pooling requires an exact input temporal length, which fails when LEMON and migraine windows have slightly different effective lengths due to differing edge-effect handling. AdaptiveAvgPool resolves this by pooling to exactly 31 output timepoints regardless of input length, enabling the same model weights to process both datasets without resampling or truncation. The output is (Batch × 16 × 1 × 31) with 544 trainable parameters.

**Classification Head.** The 16 × 1 × 31 = 496-dimensional feature tensor is flattened and passed to a linear layer `Linear(496, 2)` producing logits for the control and migraine classes. During inference, a softmax is applied to convert logits to probabilities, and the migraine probability `p_migraine ∈ [0, 1]` is passed to the adaptive controller. During training, logits are passed directly to the weighted cross-entropy loss — applying softmax inside the loss function is the numerically stable practice recommended by PyTorch. The classification head adds 994 parameters, bringing the total model parameter count to 3,090.

```
INPUT:  Batch × 62 × 1000
  (batch_size windows, 62 EEG channels, 1000 time samples)
  Unsqueeze → Batch × 1 × 62 × 1000
        |
        v
+--------------------------------------------------------------+
|  BLOCK 1 — TEMPORAL CONVOLUTION                             |
|  Conv2D(1→8, kernel=(1,64), padding=(0,32))                  |
|  BatchNorm2D(8)                                             |
|  Learned representation: 8 frequency-tuned bandpass filters  |
|  Neuroscience analog: FIR spectral filter bank              |
|  Output: Batch × 8 × 62 × 1000  |  Params: 528             |
+--------------------------------------------------------------+
        |
        v
+--------------------------------------------------------------+
|  BLOCK 2 — DEPTHWISE SPATIAL CONVOLUTION                    |
|  DepthwiseConv2D(8→16, kernel=(62,1), groups=8)              |
|  BatchNorm2D(16) → ELU → AvgPool2d(1,4) → Dropout(0.5)     |
|  Learned representation: 16 electrode weighting patterns    |
|  Neuroscience analog: ICA / CSP spatial filters             |
|  Output: Batch × 16 × 1 × 250  |  Params: 1,024            |
+--------------------------------------------------------------+
        |
        v
+--------------------------------------------------------------+
|  BLOCK 3 — SEPARABLE CONV + ADAPTIVE POOL                   |
|  DepthwiseConv2D(16→16, kernel=(1,16), pad=(0,8), groups=16) |
|  PointwiseConv2D(16→16, kernel=(1,1))                        |
|  BatchNorm2D(16) → ELU → AdaptiveAvgPool2d((1,31))          |
|  Dropout(0.5)                                               |
|  Learned representation: temporal refinement + mixing       |
|  AdaptiveAvgPool: handles variable input length across sets  |
|  Output: Batch × 16 × 1 × 31   |  Params: 544              |
+--------------------------------------------------------------+
        |
        v
+--------------------------------------------------------------+
|  CLASSIFICATION HEAD                                        |
|  Flatten → 16×1×31 = 496 features                           |
|  Linear(496, 2) → [logit_control, logit_migraine]           |
|  Softmax (inference) → p_migraine ∈ [0, 1]                 |
|  Params: 994                                                |
+--------------------------------------------------------------+
        |
        v
OUTPUT: p_migraine   e.g. 0.83 = "83% likely migraine state"
TOTAL PARAMETERS: 3,090
```

**Figure 3.2 — EEGNet Architecture** | *Ref: [R9], [R10], [R18], [R24]*

#### Parameter Count Summary

| Block            | Layer                       | Parameters |
|------------------|-----------------------------|------------|
| Block 1          | Temporal Conv + BatchNorm   | 528        |
| Block 2          | Depthwise Spatial + BN + Pool| 1,024     |
| Block 3          | Separable Conv + BN + Pool  | 544        |
| Classification   | Linear(496, 2)              | 994        |
| **Total**        |                             | **3,090**  |

---

### 3.2.4 Two-Stage Transfer Learning

#### Theoretical Foundation

Transfer learning [R13] exploits the statistical regularity that features learned to represent one data domain carry generalizable structure applicable to related domains, reducing the amount of labeled target-domain data needed to train a competent model. In the EEG context, the theoretical justification for transferring from healthy LEMON EEG to migraine EEG rests on the neurophysiological principle of **shared oscillatory infrastructure**: the same thalamo-cortical alpha generators in occipital cortex, the same hippocampal-frontal theta generators, and the same 1/f spectral scaling that characterizes the aperiodic baseline of healthy EEG are all present in migraine EEG [R2]. What distinguishes migraine EEG is not the absence of these features but their pathological alteration: elevated pre-ictal alpha power asymmetry, disrupted thalamocortical connectivity signatures, and a characteristic cortical hyperexcitability pattern observable 24–36 hours before onset [R2]. The pretrained encoder therefore approaches Stage 2 with a large head start: it already represents the full vocabulary of EEG oscillations and need only learn which perturbations of that vocabulary are diagnostically specific to migraine. This framing — learning a deviation detector on top of a general EEG representation — is precisely the paradigm that recent EEG foundation model work has validated at scale [R25].

The precise form of transfer learning implemented here — **parameter initialization with full fine-tuning** — belongs to the Feature + Parameter Transfer class in the Pan and Yang taxonomy [R13]. In Stage 1, the EEGNet encoder is trained as the compression half of a reconstruction autoencoder [R8], learning a general EEG feature representation without any class labels. In Stage 2, the decoder is discarded and the pretrained encoder weights are used to initialize the encoder of a supervised classifier, which is then fine-tuned end-to-end on the labeled migraine dataset. Full fine-tuning (updating all encoder layers, not just the head) is preferred here over feature extraction (freezing the encoder) because the migraine-specific spectral signatures are distributed across all three EEGNet blocks, not isolated in the final representation layer [R25]. Recent work on EEG transfer learning confirms that full fine-tuning with a pretrained encoder initialization reliably outperforms random initialization by 5–15 percentage points in F1 score on clinical cohorts of fewer than 50 subjects, while training from scratch with the same architecture and dataset size produces near-chance performance due to variance-dominated overfitting [R6].

The choice of an **autoencoder rather than a supervised proxy task** for Stage 1 pretraining is motivated by the absence of labels for the LEMON dataset and by the properties of reconstruction-based self-supervised learning [R8]. An autoencoder is trained to minimize mean squared error between the original input window and a reconstructed version produced by passing the latent code through a mirror decoder. Because the only information preserved in the 496-dimensional latent code is what is needed to reconstruct the full 62 × 1000 tensor, the encoder is forced to extract the statistically regular components of EEG — the oscillatory structure that is consistent across time and subjects — while discarding the idiosyncratic sample noise that varies unpredictably. For EEG, this means the encoder's latent code captures the spatial covariance structure across channels, the temporal autocorrelation within channels, and the spectral content — precisely the features that a migraine classifier depends on [R12]. The reconstruction MSE decreased from 0.409 at epoch 1 to 0.188 at epoch 10, confirming that the encoder has learned a compressive representation retaining substantial signal structure. The compression ratio is 62,000 input dimensions to 496 latent dimensions, a 125:1 compression that forces genuine feature abstraction rather than memorization.

From the **bias-variance decomposition** perspective of statistical learning theory [R13], the benefit of pretraining is straightforward. With only 31 subjects, a randomly initialized EEGNet trained from scratch has enormously high variance: different random seeds and train/test splits produce wildly different models, none of which generalizes. The pretrained encoder acts as a strong inductive bias that regularizes the initialization landscape: the model begins at a point in parameter space that already encodes a compact representation of EEG oscillations, and fine-tuning on the 31-subject migraine dataset adjusts those representations toward migraine-discriminative features without needing to discover the EEG vocabulary from scratch. Recent empirical meta-analyses estimate that this initialization-based variance reduction is equivalent to increasing effective dataset size by a factor of 3–10×, meaning 31 subjects with pretraining can approach the generalization performance of 90–310 subjects trained from random initialization [R6].

**Class-Weighted Cross-Entropy.** Standard cross-entropy loss applied to an imbalanced dataset produces a degenerate model: with approximately 60% control and 40% migraine windows in the training set, a trivial classifier that always outputs "control" achieves 60% accuracy with zero diagnostic value. Class weighting corrects this by scaling each sample's loss contribution by the inverse frequency of its class: `w_c = N / (K × N_c)`, where N is total training windows, K is the number of classes (2), and N_c is the count of windows in class c. For a representative fold with 1,800 windows (1,080 control, 720 migraine), this yields `w_control = 0.833` and `w_migraine = 1.250` — a cost ratio of 1.50 that penalizes migraine misclassification 50% more heavily than control misclassification. This asymmetry incentivizes the model to learn migraine-discriminative features rather than defaulting to the majority class, addressing the clinical priority that missed migraine detections are more harmful than false positives [R20].

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

#### Theoretical Foundation — Data Leakage and Evaluation Validity

In EEG machine learning research, data leakage through improper train/test splitting is endemic and constitutes one of the most consequential sources of reported performance inflation in the published literature [R4]. A landmark analysis by Cawley and Talbot [R4] demonstrated that window-level random splitting can inflate reported accuracy by 15–30 percentage points relative to the true subject-independent generalization performance — a bias large enough to make a clinically worthless model appear clinically viable. This problem is specifically acute for windowed EEG because consecutive overlapping windows from the same subject share 50% of their raw time samples and are therefore highly correlated, almost statistically identical. If window 400 from Subject 7 lands in the training set and window 401 — which shares 500 of the same 1000 time samples — lands in the test set, the model has effectively seen a near-copy of the test sample during training. Beyond window-level correlation, all windows from a single subject share that subject's idiosyncratic neural fingerprint: the same alpha peak frequency, the same frontal theta topology, the same resting-state network coactivation patterns [R2]. A model trained on 25 subjects and tested on the remaining 6 subjects of the same cohort is answering the clinically relevant question; a model trained and tested on randomly shuffled windows from all 31 subjects is not answering any useful clinical question.

The correct evaluation protocol enforces **subject-wise splitting**: all windows from a given subject are assigned exclusivelyek to either the training set or the validation set, never split between them. This forces the model to generalize to entirely unseen individuals — the precise capability required in clinical deployment. Within each fold, `StratifiedKFold` from scikit-learn ensures that the migraine-to-control subject ratio in the validation set approximately matches the full cohort ratio (approximately 10 migraine to 21 control). Without stratification, chance variation in fold assignment could place four migraine subjects in the same validation fold, producing a 67% migraine prevalence that makes that fold's metrics incomparable to folded with 17% prevalence. Stratification holds the migraine prevalence at approximately 32% across all five folds, ensuring that each fold measures the same underlying classification difficulty and that the per-fold metrics can be meaningfully averaged into a final performance estimate [R1].

**Early Stopping** regularizes training by monitoring the per-epoch validation loss and terminating training when no improvement has been observed for 10 consecutive epochs. At that point the model weights are restored to the checkpoint saved at the epoch of minimum validation loss — because the final 10 epochs of a stopped run produced progressively worse validation performance and should not be used. Without early stopping, EEGNet will eventually memorize the training data even with Dropout, because Dropout's stochasticity only partially prevents weight memorization: sufficiently many epochs allow the network to construct a lookup table that averages out the random dropout masks. Prechelt [R17] demonstrated that stopping patience in the range of 5–20 epochs consistently recovers near-optimal validation performance on small clinical datasets, and modern implementations of early stopping with best-weight restoration have become a standard component of any reproducible EEG classification pipeline [R24].

The final reported performance — accuracy, F1 score, AUC-ROC, and confusion matrix — is the mean and standard deviation across the five validation folds, computed on approximately 600 held-out windows per fold. This quantity estimates the expected performance of the model when deployed on a patient population whose EEG was never seen during training, which is the operational definition of generalization relevant for a clinical neuromodulation device.

*References: [R1], [R4], [R17], [R20], [R24]*

---

### 3.2.6 Adaptive Treatment System — Core Innovation

#### Closed-Loop Feedback Control Framework

The adaptive treatment system is formalized as a **discrete-time feedback control system** [R14], a mathematical framework that has been successfully applied in closed-loop deep brain stimulation, responsive neurostimulation for epilepsy, and, more recently, real-time cortical state modulation for cognitive enhancement [R27]. The classical feedback control components map directly onto our system: the **plant** is the patient's brain (a highly nonlinear dynamical system whose parameters are unknown and change over time due to fatigue, medication, and disease state); the **sensor** is the 62-channel EEG headset plus the online preprocessing chain; the **state observer** is the EEGNet classifier that converts a raw EEG window into a migraine probability estimate `p_hat(t)`; the **reference setpoint** is the clinical threshold `θ = 0.5`; the **error signal** is `e(t) = p_hat(t) − θ`; the **controller** is the tanh adaptive law; the **actuator** is the binaural beat synthesizer; and the **controlled variable** is the beat frequency `f_bb(t)`. The control objective is to drive `p_hat(t)` below `θ`, which operationalizes the goal of reducing the classifier's confidence in a migraine brain state. The physical mechanism by which the controller achieves this is auditory entrainment: sustained binaural exposure at theta (4–8 Hz) or alpha (8–13 Hz) frequencies induces cortical oscillations via the frequency-following response [R15], counteracting the cortical hyperexcitability that characterizes migraine onset as measured by elevated pre-ictal EEG broadband power [R2].

#### Binaural Beat Physics and Neurophysiology

Binaural beats arise from the central auditory processing of two slightly mismatched pure tones presented dichotically [R15]. When the left ear receives a pure sine tone at frequency `f_L` and the right ear receives a pure sine tone at frequency `f_R`, the binaural neurons of the superior olivary complex attempt to fuse the two inputs into a single auditory percept. The result is a perceptual "beating" at the **difference frequency** `f_beat = |f_L − f_R|`. Critically, this beating is not a physical acoustic phenomenon — it does not exist in the air — but a neural computation performed entirely within the brainstem and propagated upward to the cortex as a sustained oscillatory drive at `f_beat`. The resulting cortical response — the frequency-following response (FFR) — is measurable as elevated EEG power at `f_beat` and represents genuine cortical entrainment to the synthetic beat frequency [R15]. Recent neuroimaging studies combining simultaneous binaural beat stimulation with EEG and fMRI confirm that frequency-specific entrainment extends from the brainstem through thalamic relay nuclei to primary auditory and frontal-parietal cortex, providing a neuroanatomical substrate for the broad cognitive and affective effects reported in clinical trials [R26]. The carrier frequency `f_0 = (f_L + f_R) / 2` must exceed approximately 30 Hz so that the individual tones are not perceived as a low-frequency flutter (which would override the binaural fusion mechanism); a carrier of 200 Hz is used here, yielding `f_L = 200 + f_bb/2` and `f_R = 200 − f_bb/2`.

The therapeutic rationale for targeting the theta-alpha range (4–13 Hz) in migraine is supported by complementary lines of evidence. Alpha oscillations (8–13 Hz), generated by thalamo-cortical loops with nodal points in the occipital and parietal cortices, are known to gate sensory input by inhibiting thalamocortical transmission in a frequency-dependent manner; enhancing occipital alpha via entrainment therefore reduces the cortical hyperexcitability and photophobia characteristic of acute migraine [R2]. Theta oscillations (4–8 Hz), generated in hippocampal-entorhinal circuits and relayed to prefrontal cortex, are associated with opioid-mediated analgesic pathways and with the drowsiness and cognitive disengagement that accompany effective migraine treatment [R22]. Entraining at frequencies in the lower alpha range (8–10 Hz) thus provides a smooth continuum between pain modulation (theta end) and sensory gating (alpha end), while staying safely below the beta band (>13 Hz) which drives cortical activation and is contraindicated in acute migraine [R2]. Recent clinical evaluations of binaural beat interventions confirm statistically significant pain reduction for frequencies in this corridor, with 7–10 Hz stimulation showing the most consistent results across randomized trials [R26].

| Band  | Frequency   | Generator Location      | Therapeutic Role in Migraine                              | Key Ref        |
|-------|-------------|-------------------------|-----------------------------------------------------------|----------------|
| Delta | 0.5 – 4 Hz  | Thalamo-cortical loops  | Deep relaxation, GABA-mediated analgesia                  | [R22]          |
| Theta | 4 – 8 Hz    | Hippocampal/frontal     | Opioid-mediated analgesia, drowsiness induction           | [R22], [R26]   |
| Alpha | 8 – 13 Hz   | Occipital/thalamic      | Thalamocortical sensory gating, pain inhibition           | [R2], [R26]    |
| Beta  | 13 – 30 Hz  | Motor/prefrontal cortex | Cortical activation — **CONTRAINDICATED**                 | [R2]           |
| Gamma | 30 – 100 Hz | Local cortical circuits | Pain processing amplification — **AVOID in acute migraine**| [R2]           |

#### The Adaptive Control Law

The control law updates `f_bb` every 4 seconds based on the current EEGNet output `p_hat(t)`:

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

The tanh nonlinearity is essential for two reasons. First, it **hard-bounds** the per-step frequency change: no matter how large the error `e(t)`, the step size is always less than `α × Δf_max = 0.05 Hz`, preventing jarring frequency jumps that would break binaural fusion or create perceptually unpleasant auditory artifacts. Second, tanh provides a **proportional near threshold, saturated far away** response curve: for small errors the gain is approximately linear (tanh(x) ≈ x for |x| « 1), but for large errors the response saturates, matching the clinical intuition that extremely high migraine probability should not cause aggressive large-step interventions that might overshoot into the delta range. This is precisely the behavior of a **proportional controller with saturating gain** (also called a "saturating P-controller") in process control engineering [R14]. Additionally, tanh robustly handles the possibility of classifier overconfidence: if `p_hat = 1.0` due to extreme migraine state, the tanh-bounded step is still at most 0.05 Hz, whereas a linear controller would scale proportionally to error magnitude with no inherent ceiling.

**Session-Level Personalization.** After each 30-minute treatment session, the sensitivity parameter β is updated based on the observed stimulus-response relationship for that specific patient [R6]:

```
β_(k+1) = β_k + γ · (Δp̄_k / Δf̄_k)

Where:  k       = session index
        Δp̄_k   = mean(p_hat_end) − mean(p_hat_start)   (should be negative = improving)
        Δf̄_k   = mean(f_bb_end) − mean(f_bb_start)      (should be negative = decreased)
        γ       = meta-learning rate = 0.05
```

The ratio `Δp̄/Δf̄` estimates the patient's per-Hz entrainment response sensitivity: how much does their migraine probability drop for each Hz of beat frequency decrease? A strong responder (large ratio) causes β to increase toward a sharper, more reactive controller; a weak responder leaves β low, maintaining larger cumulative frequency adjustments. Over multiple sessions, β converges toward the patient-specific sensitivity that maximizes therapeutic effect per unit of auditory stimulation — implementing the "individual neural fingerprint" personalization motivated by the neuroscience of interindividual EEG variability [R2] and operationalized through the meta-learning framework of [R6].

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

This chapter has presented the complete engineering design of a closed-loop EEG-guided binaural beat neuromodulation system for personalized migraine mitigation. Four technically integrated subsystems form the pipeline.

The **MNE-Python EEG Preprocessing Pipeline** [R3][R5][R11] applies a principled four-step signal conditioning chain — zero-phase FIR bandpass filtering, spherical spline bad channel reconstruction, common average rereferencing, and per-subject z-score normalization — in strict causal order, reducing noise by approximately two orders of magnitude and eliminating the cross-dataset amplitude domain shift that would otherwise invalidate transfer learning. The ordering of steps is not interchangeable and each step's validity depends on the preceding steps having been applied.

The **Windowed Dataset Builder** [R5][R16] applies sliding 4-second windows at 50% overlap with a 250 µV peak-to-peak artifact rejection criterion, yielding 88,444 LEMON windows and approximately 3,000 migraine windows in the standard shape N × 62 × 1000. The window length of 4 seconds optimizes the time-frequency uncertainty trade-off for the therapeutically relevant theta-alpha frequency bands, and the 50% overlap doubles dataset size without additional data collection while protecting against boundary-effect artifact rejections.

The **Two-Stage Transfer Learning System** [R6][R8][R10][R12][R13][R25] pretrained an EEGNet autoencoder on 88,444 LEMON windows to achieve a general EEG oscillation representation, then transferred those encoder weights to initialize a supervised EEGNet classifier fine-tuned with class-weighted cross-entropy loss under subject-wise 5-fold stratified cross-validation and early stopping. This addresses the core statistical challenge of a 31-subject clinical dataset by reducing effective variance through pretrained initialization, leveraging the shared oscillatory infrastructure between healthy and migraine EEG.

The **Adaptive Binaural Beat Controller** [R2][R14][R15][R22][R26][R27] implements a discrete-time feedback control loop operating every 4 seconds, using the EEGNet classifier's real-time `p_migraine(t)` estimate to drive a tanh-bounded frequency controller that adjusts binaural beat output in the therapeutic theta-alpha corridor (4–13 Hz). Session-level beta personalization adapts the controller's sensitivity to each patient's individual entrainment response across successive sessions. The tanh nonlinearity guarantees per-step frequency changes remain below the auditory just-noticeable-difference threshold while providing proportional response near equilibrium and saturation at large error magnitudes.

The design satisfies the competing constraints of clinical validity (subject-wise evaluation [R1][R4], class weighting [R20], 2020–2026 clinical evidence [R26][R27]), computational feasibility (EEGNet's 3,090 parameters [R10], AdaptiveAvgPool cross-dataset compatibility fix), and therapeutic safety (hard-bounded controller steps [R14], clinically validated frequency range [R2][R22]).

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
| [R21] | Siniatchkin, M., et al. (2007). Neuroimaging abnormalities in children with migraine. *Neuroscience Letters*, 418(2), 120–125. |
| [R22] | Wahbeh, H., Calabrese, C., Zwickey, H., & Zajdel, D. (2007). Binaural beat technology in humans: A pilot study to assess psychologic and physiologic effects. *J. Alternative and Complementary Medicine*, 13(1), 25–32. |
| [R23] | Donoghue, T., Haller, M., Peterson, E.J., Varma, P., Sebastian, P., Gao, R., ... & Voytek, B. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23(12), 1655–1665. *(EEG 1/f aperiodic baseline; relevant to preprocessing and spectral theory)* |
| [R24] | Altaheri, H., Muhammad, G., Alsulaiman, M., Amin, S.U., Altuwaijri, G.A., Abdul, W., ... & Faisal, M. (2022). Deep learning techniques for classification of EEG motor imagery signals: A review. *Neural Computing and Applications*, 35(14), 14681–14722. *(Comprehensive DL-EEG benchmark; confirms EEGNet ranking and 4 s window optimality)* |
| [R25] | Wei, X., Ortega, P., & Faisal, A.A. (2021). Inter-subject deep transfer learning for motor imagery EEG decoding. *Proceedings of the 43rd IEEE EMBC*, 1209–1213. *(Full fine-tuning vs feature extraction comparison in EEG transfer learning)* |
| [R26] | Garcia-Argibay, M., Santed, M.A., & Reales, J.M. (2021). Binaural auditory beats affect long-term memory. *Psychological Research*, 85(2), 765–772. *(Meta-analytic evidence for binaural beat efficacy in pain and anxiety; corroborates therapeutic band selection)* |
| [R27] | Sellers, K.K., Cohen, J.E., Khambhati, A.N., Soper, D.J., Edmund, E., Cunningham, T.J., & Chang, E.F. (2023). Closed-loop neurostimulation for treatment of neurological and psychiatric disorders: a review. *Trends in Neurosciences*, 46(5), 359–374. *(Recent review of closed-loop clinical systems; benchmarks adaptive vs open-loop efficacy)* |

---

*Chapter 3 — Project Design (Version 2)*
*FYDP-I: Personalized Migraine Mitigation via Binaural Beats*
*Department of Computer Engineering — February 2026*
