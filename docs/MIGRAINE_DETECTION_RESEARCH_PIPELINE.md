# EEG-Based Migraine Detection & Personalized Binaural-Beat System — Research Pipeline Design

**Scope:** Scientific feasibility analysis and end-to-end research pipeline for binary migraine-vs-control detection from the Chamanzar et al. (2020) ultra-high-density EEG dataset, and its downstream integration with a personalized binaural-beat intervention.

**Dataset:** "Ultra high-density EEG recording of interictal migraine and controls: sensory and rest" (128-channel, 17 migraine / 18 matched controls, recordings during SSAEP, SSVEP, and rest). Collected Jan–Dec 2018.

**Before reading:** This document is a research design, not implementation code. It references the existing project artifacts:
- `Migraine_Dataset_GPU_Preprocessing.ipynb` (existing MNE pipeline)
- `data/MIGRAINE_GPU_preprocessed/` (existing outputs)
- `models/training_summary.txt` (prior EEGNet transfer-learning attempt: **AUC ≈ 0.46 — chance level**)
- `docs/ADAPTIVE_CONTROL_LAW_EXPLANATION.md` (existing binaural-beat control law)
- `Dataset/Migraine_Control_Demographics.xlsx` (36 subjects: 18 M, 18 C)

---

## 1. Executive Summary

1. **The dataset is usable, but only for a narrowly-defined task.** It supports a *trait-level* ("has migraine disorder") interictal classifier with modest, honestly-reported performance expectations. It **cannot** support detection of an *ongoing migraine attack* (ictal state), because every recording is interictal. Referring to this as "migraine Yes/No" is scientifically defensible only if "Yes" means "this person has a migraine disorder based on interictal EEG" — not "this person is currently having a migraine."

2. **The biggest scientifically actionable signal in this specific dataset is likely NOT resting broadband power.** The original publication (Chamanzar et al., 2020) reported **abnormalities in cortical patterns of coherence** during rest and sensory stimulation. A pipeline that only stacks resting 2-second epochs (as in the current `data/` folder) discards two of the three recording conditions and all channel-by-channel phase relationships — the exact information the paper identified as abnormal.

3. **The prior EEGNet attempt failed at chance level for identifiable, fixable reasons:** (a) it projected the 130-channel migraine data down to the 62-channel LEMON layout, destroying the HD-EEG advantage; (b) it used only resting broadband epochs; (c) the intended 250 Hz resampling was never applied in the GPU notebook (epochs are 1024 samples ≈ 2 s @ 512 Hz); (d) label files were derived from folder prefixes, so medicated/excluded patients (M2, M6, M18) were silently included as "migraine"; (e) with ~30 subjects, subject-leakage-free deep learning has very wide error bars, and chance-level performance is consistent with Varoquaux (2018) on small samples.

4. **15–30 s windowing is appropriate**, with strict rules: windows must never cross validation splits, and subject-level aggregation must be the final prediction unit.

5. **GAN augmentation is a last resort.** The literature is clear (Lashgari et al., 2020) that classical augmentation, transfer learning, and self-supervised pretraining should precede GAN synthesis for EEG. With ~30 subjects, a GAN trained on this dataset alone will overfit; a GAN pretrained on large public EEG corpora is more defensible.

6. **The binaural-beat system requires two distinct models**: (i) a **trait classifier** (this pipeline: migraine vs control; used for screening/personalization initialization) and (ii) an instantaneous **state estimator** (band-power deviations from the individual's baseline; used in the closed-loop control law). They must not be conflated.

---

## 2. Is This Dataset Suitable for Migraine Yes/No Detection? — Question 1

### What speaks in favor
- **HD-EEG with 128 channels** is at the high end of what exists publicly; most public migraine EEG datasets are 8–32 channel or clinical montage.
- **Three well-controlled conditions** per person (rest, SSAEP 4/6 Hz, SSVEP 4/6 Hz), with demographic matching (age/gender) built into the study design — matching reduces confounds in the label definition.
- **Stimulus frequencies are known** (4 and 6 Hz), which enables precise SSVEP/SSAEP amplitude and harmonic analysis — a well-established migraine-relevant signal (altered visual/auditory driving).
- **The original paper demonstrated group-level EEG differences** in this exact data (coherence abnormalities). A classifier that targets those same features starts from a known effect.

### What speaks against
- **Only ~36 subjects (18 migraine / 18 control)**; after the original exclusions (M13 — no SSAEP; M2, M6, M18 — medication) and after reserving a held-out test cohort, you have at most ~14–17 migraine vs ~18 control subjects for development. This is a **pilot/feasibility sample**, not a statistically powered clinical trial.
- **Interictal only** (see Section 3). The population-level effects are subtle; single-subject single-session discrimination is at the edge of what the signal supports.
- **Recording-level confounds** (session length differences, slight montage differences between subjects, the `P15`/`M15` naming glitch) require careful QC.
- **No external validation cohort** with a compatible 128-channel montage exists publicly; generalizability claims must be explicitly limited.

### Verdict
**Conditionally suitable for a proof-of-concept interictal trait classifier**, provided the goal is framed as "detect the migraine brain-state trait from resting and sensory EEG" and the evaluation uses subject-level, leakage-free validation with honest error bars. It is **not suitable** for an ictal-attack detector, for a regulatory-grade diagnostic, or for claims of wearable-device performance without a separate low-density EEG study.

---

## 3. The Interictal vs Ictal Problem — Question 2 (the defining limitation)

### The scientific facts
- **Ictal (during-attack) recordings** of migraine are extremely rare in public datasets, and this dataset contains none. All BDFs were recorded interictally (between attacks).
- Interictal EEG differences between migraineurs and controls are documented but **subtle and frequency-specific**: altered alpha peak frequency and alpha power, increased theta, and — most prominently — **reduced habituation / abnormal evoked responses and abnormal coherence** (Bjørk et al., 2009, systematic review of quantitative EEG in migraine; Coppola et al., 2009, on habituation; Chamanzar et al., 2020, on coherence).
- The brain state during an attack is physiologically distinct (altered autonomic state, pain-network activity, photophobia-related network changes), so **a model trained on interictal data cannot be claimed to detect an ongoing attack.** That is not a modeling failure; it is a domain-shift limit that no architecture can overcome without ictal training data.

### How to frame the task honestly
| Claim | Supported? |
|---|---|
| "Model detects whether a person's interictal resting/evoked EEG resembles migraine vs control" | ✅ Supported by this dataset |
| "Model detects whether a person is currently having a migraine attack" | ❌ **Not supported** — requires an ictal cohort (ambulatory wearable recordings during attacks) |
| "Model can be used for personalized therapy initialization" | ✅ Supported as a research pilot (trait → personalization) |

### Consequence for your project
- **Rename/reframe the deliverable**: "Interictal migraine trait detection from HD-EEG" as the research claim; "real-time attack monitoring" as a future, separate study requiring new data collection (e.g., home-use wearable EEG logged during the attack-prodrome–ictal–postdromal cycle).
- For the binaural-beat system, the interictal classifier is still useful: it enables **personalization** (who is in the target population, what is their baseline deviation profile) without needing to detect an attack in progress.

---

## 4. Window Length: Is 15–30 s Appropriate? — Question 3

### Yes, with conditions
- EEG is commonly treated as quasi-stationary over windows of a few seconds to tens of seconds (Cohen, 2014). Windows of 15–30 s are standard in EEG-stationarity-based classification and are long enough for:
  - reliable PSD estimation at low frequencies (e.g., delta 0.5–4 Hz needs ≥ 2–4 s of data for stable estimates; 15–30 s gives many Welch segments);
  - meaningful alpha-peak tracking (8–13 Hz) and
  - a spectrogram with adequate time–frequency resolution (e.g., 2–4 s FFT windows with 50–75% overlap inside each 15–30 s window).

### Rules that must accompany windowing
1. **No cross-validation leakage through windowing.** All windows from one recording must belong entirely to one split (subject-level grouping; see Section 8).
2. **Overlap is acceptable within a split** (it increases samples and acts as augmentation), but **never across split boundaries**, and overlaps must be reported (effective duplication inflates confidence if unaccounted).
3. **Window-level predictions must be aggregated to subject level** (majority vote or mean probability) for the final reported metric. Per-window accuracy overstates performance because windows within one subject are correlated.
4. **Recommended scheme**: 20 s windows with 10 s stride (50% overlap) → resting/SSVEP/SSAEP each yield ≈ (recording duration − 20) / 10 + 1 windows. For a typical 200 s resting recording this is ≈ 19 windows/subject/condition → roughly 34 subjects × ~19 × 3 conditions ≈ **~1,900–2,000 windows** — a defensible training pool for a *feature-based* classifier, marginal for *deep* learning.
5. **Event-locked caution**: for SSVEP/SSAEP, prefer windows aligned to stimulus blocks rather than arbitrary cuts. The 4 Hz vs 6 Hz stimulus blocks carry distinct information (different driving frequencies, different habituation slopes) — the windowing scheme should preserve this (see Section 6).

---

## 5. EEG Preprocessing & Artifact Removal — Question 4

The existing MNE pipeline in `Migraine_Dataset_GPU_Preprocessing.ipynb` is a reasonable backbone. The following changes make it research-grade:

### 5.1 Fix known correctness bugs in the current pipeline (found during review)
| Issue | Fix |
|---|---|
| `TARGET_SFREQ = 250` is defined but **never applied** — saved epochs are 1024 samples ≈ 2 s @ 512 Hz | Actually resample to 250 Hz (or keep 512 Hz, but decide once and record it in a data dictionary). A single canonical `sfreq` is required for consistent spectrogram/feature code. |
| **C12 SSAEP/SSVEP failed** ("can't filter empty Epochs") — all epochs rejected | Add a guard: if `len(epochs) == 0` after rejection, save an explicit `EMPTY` marker and continue; investigate why ICA/reference left no clean epochs. Also consider relaxing the amplitude rejection for non-rest conditions where stimulation artifacts are physiological, not bad data. |
| **M2, M6, M18 (medicated) and M13 (no SSAEP) are silently included** via folder-prefix labels | Build labels from `Migraine_Control_Demographics.xlsx` (aura status, medication status) and run the primary analysis on the original exclusion set; run a sensitivity analysis with the extras. |
| **Channel mismatch risk for transfer learning** (130 vs 62 channels, LEMON) | Never down-project HD-EEG to a 62-channel layout blindly. If transfer learning from LEMON is used, either (a) select the intersection montage and re-derive, or (b) use **channel-agnostic** self-supervised encoders. |
| **`filename.startswith(f'{subject_id}_')` prefix bug** in the stacking code matches `C1` → `C10_...`, `C11_...`, etc. | Use exact subject matching (`^C1_` with word boundary). (The `.npz` was never actually written, but the bug is latent in the windowing step you will build next.) |
| **M1/M2 vs average referencing varies per subject** | Pick **one reference policy** for all subjects (e.g., average reference after interpolation; M1/M2 only if both are clean and of good quality). Reference consistency is essential for connectivity/coherence features. |
| **`annotate_bad_segments` uses raw peak-to-peak only** | Keep it as a coarse gate, but drive the main rejection with ICA + ASR, not threshold-only. |

### 5.2 Recommended preprocessing chain (research-grade)
1. **Load & inspect**: `read_raw_bdf`, check native sfreq and channel count (expect ~130: 128 EEG + M1/M2 + EOG/ECG/aux). Record which channels are actually present per file (they vary).
2. **Channel set cleanup**: drop non-EEG (GSR, Erg, Resp, Plet, Temp, Status) as now; remap EOG (LO1/LO2/IO1/IO2/SO1), ECG; keep M1/M2 for reference options.
3. **Reference**: average reference (after excluding bad channels), consistent across all subjects; store `mastoid` as an alternative only for sensitivity analysis.
4. **Filtering**: 1–100 Hz bandpass (fir, zero-phase) + 50/100 Hz notch (as now). If ictal-state estimation is ever added later, a narrower 1–40 Hz band is safer for wearable devices; for this analysis, keep 1–100 Hz to preserve gamma.
5. **Artifact removal, tiered**:
   - **ASR** (Mullen et al., 2015) for high-amplitude movement/electrode artifacts — *carefully*, with a conservative cutoff (e.g., 20–30 SD), applied per recording before ICA;
   - **ICA (Infomax extended)** with **ICLabel** (Pion-Tonachini et al., 2019) for eye/muscle/heart components — already in the pipeline (good). Keep the ≥0.80 threshold but validate against manual labels on a QC subset;
   - **Channel interpolation** as now, but log every interpolated channel and never interpolate > ~10–15% of channels.
6. **Segment-level rejection**: amplitude threshold (§250 µV) only after ICA; record keep-rates per subject/condition (already in `preprocessing_summary.csv` — good).
7. **QC report per subject**: number of channels kept, rejected ICs, rejected segments, and a PSD sanity plot. This is required for the paper's methods section and for defending exclusions.

---

## 6. Features & Time–Frequency Representations — Questions 5–6

### 6.1 Which features the literature actually supports for interictal migraine

| Feature family | Evidence | Usability on 128-ch | Primary candidate? |
|---|---|---|---|
| **Coherence / functional connectivity** (wPLI, PLV, imaginary coherence) | **The original paper's main finding** (Chamanzar et al., 2020): abnormal cortical coherence patterns in migraine | ✅ HD-EEG is ideal for graph/connectivity features | ⭐ **Primary** |
| **Band power / relative PSD** (delta, theta, alpha, beta, gamma) | Systematic review evidence of alpha slowing, theta increases (Bjørk, 2009) | ✅ Easy, robust | ⭐ **Primary (coarse)** |
| **Alpha peak frequency** | Slowed alpha peak in migraine (Bjørk, 2009) | ✅ 1 scalar/channel | High value, cheap |
| **SSVEP/SSAEP amplitude & harmonics at 4/6 Hz** | Altered visual/auditory driving & habituation (Coppola et al., 2009) | ✅ High | ⭐ **Primary for task conditions** |
| **Habituation slope** across repeated stimulus blocks | Well-replicated dyshabituation in migraine | ✅ Needs block-aligned epoching | ⭐ **High value** |
| Spectral entropy / sample entropy | General EEG-complexity literature | ✅ | Secondary |
| Hjorth parameters | Cheap, classic | ✅ | Secondary |
| Whole-electrode spectrogram images for CNN | DL-EEG literature (Roy et al., 2019) | ✅ but 130 channels → 130 × T × F tensor | Secondary (see §6.3) |0
| Raw time-series for CNN (EEGNet-style) | Lawhern et al. (2018) | ✅ | Baseline only |

### 6.2 Let the dataset's own paper guide you
The single most important scientific guideline: **the effect your target paper found is in coherence during rest and sensory stimulation.** A competing pipeline that uses only single-channel spectral power on resting 2-s epochs is discarding the highest-signal information. Your feature matrix should therefore be built around:
- **Per-condition, per-band connectivity matrices** (e.g., 130×130 wPLI in delta/theta/alpha/beta/gamma, or a rois-reduced 64×64 grid),
- **Band-power topographies** (130 channels × 5 bands → treat as small 2D images),
- **Alpha peak frequency distribution** across channels,
- **Stimulus-locked metrics** for SSAEP/SSVEP: amplitude at 4/6 Hz and harmonics, inter-trial coherence, habituation slope across blocks.

### 6.3 Spectrograms vs raw EEG vs PSD vs wavelets vs connectivity — a decision framework

| Representation | Strengths | Weaknesses for THIS dataset | Recommendation |
|---|---|---|---|
| **Spectrograms (STFT) via CNN** | Captures time×frequency; natural for DL; amputation of interpretability | 130 channels × (time × freq) is huge → needs channel average/topomap reduction; inter-channel phase is lost unless stack is 3D; small-n overfitting | **Use as one of two model views**, not the only view; constrain with channel subsets (frontal, temporal, occipital) or 2D topo-stack images |
| **Raw EEG + compact CNN (EEGNet/ShallowConvNet)** | Preserves full temporal+spatial structure; minimal assumptions | Needs many subjects; the prior attempt failed at chance with this exact approach | Keep as an honest **baseline**, but do not expect it to win at n≈30 |
| **PSD/relative band power + classical ML** | Low variance, interpretable, robust at small n | No phase/connectivity; no fine time resolution | ⭐ **Primary model input #1** |
| **Wavelet/scalogram images** | Good time-frequency localization; multi-scale | Same DL small-n problem as spectrograms; marginal benefit over STFT for fixed bands | Optional alternative view; use CWT only if spectrogram underperforms |
| **Connectivity (wPLI/coherence) graph features** | Directly targets the paper's finding; low-dim; interpretable | Requires careful artifact handling (phase distortions) | ⭐ **Primary model input #2 — pair with PSD** |
| **Hybrid feature fusion + shallow model** | Combines best of all; robust at small n | Engineering effort | ⭐ **Recommended winner for the pilot** |

**Bottom line: do NOT bet the project on spectrogram-only deep learning.** At n ≈ 34 subjects, deep classifiers are high-variance and will almost certainly not beat a well-built feature-based (PSD + connectivity + SSVEP/SSAEP) shallow classifier. Use both: (A) **classical feature pipeline** for the primary claims; (B) **CNN on spectrograms/topomaps** as a secondary, hypothesis-generating arm — and report both with identical subject-level CV.

### 6.4 Multi-condition design
Build **three condition-specific feature sets** (resting, SSAEP, SSVEP) and fuse them:
- Level 1: per-condition classifier (does resting EEG alone discriminate? Does SSVEP driving alone?),
- Level 2: early fusion (concatenate z-scored features across conditions) or late fusion (average per-window probabilities) — **late fusion is safer and more interpretable**.
- This also tells you which condition is most informative — a genuinely interesting scientific output.

---

## 7. GAN-Based Augmentation: Justified? — Question 7

### The honest answer
**Not as a first or second resort.** The evidence base for GAN-augmentation in EEG is real but immature (Lashgari et al., 2020 review of EEG data augmentation; Hartmann et al., 2018 EEG-GAN), and the failure mode is severe with your sample size: a GAN trained on ~34 subjects will memorize the training subjects, and synthetic samples will not generalize; subject-leakage through generative models is notoriously easy to introduce accidentally.

### Order of operations (must be followed in the paper)
1. **Classic augmentation inside the split** (always): segment overlaps, time shifts, small Gaussian sensor noise (µV-scale), amplitude scaling, channel dropout (few %). These are cheap and have known, controllable effects.
2. **Mixup / segment-mixing variants** (mix between windows of the *same class and different subjects within the training fold only*): simple, effective for DL-EEG, no subject leakage if restricted to training folds.
3. **Transfer learning / self-supervised pretraining on large public EEG** (LEMON, TUH, SEED, Sleep-EDF): contrastive encoders (Banville et al., 2021; Kostas et al. — BENDR, 2021) give subject-agnostic EEG representations that transfer to small labeled sets. This is strictly better than GAN augmentation for your problem because it brings in *real* diversity from *real* data.
4. **GAN/VAE-based synthesis** — only if, after 1–3 with honest subject-level CV, you still need more samples, and then with these constraints:
   - **Never train/validate the GAN on the held-out test subjects.** The generative model may only see training-fold data; the test fold must remain unseen by both generator and discriminator.
   - Report AUC with and without synthetic data to show that gains are not leakage artifacts.
   - Report the **effective sample size** accounting for synthetic data correlation.
   - Prefer **conditional GANs conditioned on class and (if used) condition** to preserve class balance.
   - Balance classes by **class-weighted loss / stratified sampling** rather than by generating only the minority class.

**Alternative with higher scientific payoff:** instead of generating synthetic EEG, invest in (a) transfer learning from large public corpora and (b) **subject-level feature engineering** that reduces the effective dimensionality (connectivity matrices + band powers), so your n≈34 is enough. Deep generative augmentation at n=34 is more likely to produce overfit confidence than real gains.

---

## 8. Preventing Subject Leakage — Question 8

### The non-negotiable rules
1. **Split by subject, never by epoch.** All windows from one recording/subject live in exactly one fold. This is the most-cited methodological failure in medical ML (Varoquaux & Cheplygina, 2022) and it directly invalidated several EEG studies.
2. **Use grouped (group-shuffle) stratified 5-fold CV** at the subject level, **or Leave-One-Subject-Out (LOSO)** for the final evaluation. With 34 subjects, **LOSO is feasible and maximally honest** (34 models). Report both the LOSO aggregate and the 5-fold subject-grouped curves.
3. **Nested CV for any hyperparameter or model selection**: inner loop chooses hyperparameters on training folds; outer loop reports the generalization estimate. Never tune on the test fold.
4. **Final held-out cohort**: if possible, set aside 3–4 subjects (2 migraine, 2 control) that are never touched during development, then run the model once at the very end. With 34 subjects this is expensive but it is the only way to claim unbiased accuracy.
5. **Chance-level calibration**: with 2 classes and ~34 subjects, report the **chance-level accuracy bound** and its confidence interval using the method of Combrisson & Jerbi (2015) (theoretical chance with binomial confidence), because small samples routinely produce "above chance" results by luck (as the prior 5-fold results show: fold 3 AUC = 0.08, fold 4 = 0.22 — extreme variance is expected at this n).
6. **Subject-level aggregation for the final metric**: report AUC / accuracy computed on **subject-level votes** (window probabilities averaged), not on window-level predictions. Window-level metrics are inflated by within-subject correlation.
7. **Report per-fold distributions** (as the prior CSV does) with mean ± SD **and individual fold values** — the extremes are diagnostic (and were, in the prior attempt, a red flag).
8. **Data preprocessing leakage audit**: ICA, scaling, feature normalization, and any imputation must be fitted on train folds only. Standardize with fold-specific scalers. This is frequently overlooked (Varoquaux & Cheplygina, 2022).

---

## 9. Recommended Model Architectures & Validation Strategy — Question 9

### 9.1 Model ladder (from simplest to most complex)
Choose the simplest model that performs well; end the paper with the best one and report all:

| Tier | Model | Input | Why |
|---|---|---|---|
| **1 — Classical baseline** | Logistic Regression (L2), Random Forest, or gradient boosting on PCA-reduced features | PSD relative band power + wPLI connectivity + alpha-peak + SSVEP/SSAEP features | Low variance, interpretable SHAP values, robust at n≈34. **Expected to be the strongest pilot performer.** |
| **2 — Shallow DL** | ShallowConvNet / EEGNet (Lawhern et al., 2018; Schirrmeister et al., 2017) | Raw or lightly denoised 130-ch (or channel-reduced) epochs | Baseline for DL; the prior EEGNet attempt was this tier and failed — treat it as a **comparator**, not the expected winner. |
| **3 — Spectrogram/topomap CNN** | 2D CNN / ResNet-style on STFT or wavelet topomap stacks | Spectrogram images (channel-reduced or per-region) | Secondary arm; report clearly. |
| **4 — Connectivity-graph model** | Graph neural network over channel graphs, or precomputed connectivity vectors + MLP | 130×130×bands wPLI | Interesting if connectivity features win in tier 1; GNN adds capacity but risk of overfit. |
| **5 — Transfer/self-supervised** | Frozen or fine-tuned contrastive EEG encoder (trained on LEMON/TUH/etc.; Banville et al., 2021; Kostas et al., 2021) + linear probe | Raw 62-ch or channel-selected EEG | Best modern strategy for small labeled sets; the encoder brings in subject-agnostic real-data diversity without GAN leakage. |

### 9.2 Prescription for the pilot
- **Primary endpoint:** subject-level ROC-AUC (and accuracy/precision/recall/F1/sens/spec at the Youden threshold) under LOSO and 5-fold grouped CV, with 95% bootstrap CIs.
- **Primary models:** Tier 1 (logistic/GBM on PSD + connectivity + SSVEP/SSAEP features) and Tier 5 (linear probe on a public-EEG contrastive encoder).
- **Secondary/confirmatory:** Tier 2 EEGNet and Tier 3 spectrogram CNN — reported with identical CV so the paper can honestly compare.
- **Condition ablation:** report AUC per condition (rest only, SSAEP only, SSVEP only) and combined — this is a publishable scientific result in itself.
- **Aura stratification:** stratify folds by aura status (7 aura / 4 non-aura / 7 unknown among 18 M after exclusions) so one aura subgroup does not drive the result.
- **Confound checks:** verify age/gender balance across folds; report whether performance survives covariate-matching (propensity-style analysis optional at this n).
- **Statistical rigor:** per Combrisson & Jerbi (2015) and Varoquaux (2018) — report chance-level CIs and acknowledge that with n≈34 the 95% CI on AUC is wide (±0.1–0.15). A pilot with AUC ≈ 0.75–0.85 and wide CI is a *positive* result at this n; AUC 0.90 with n=34 should be treated with suspicion of leakage.

### 9.3 Diagnosis of the prior attempt (why it failed — and why it is still useful)
The prior EEGNet transfer attempt (models/training_summary.txt) is actually a **valuable negative result** that should be reported. Likely contributors, in order:
1. **Channel mismatch (130 → 62):** down-projecting destroyed the HD coherence information.
2. **Resting-only, short epochs (2 s / 1024 samples):** the known strong effects in this dataset are coherence and stimulus-driven, not resting broadband power.
3. **n≈30 with subject-grouped 5-fold:** enormous fold variance (fold AUCs 0.08–0.76) is exactly what Varoquaux (2018) predicts at this sample size.
4. **Label contamination:** excluded/medicated patients (M2, M6, M18) present as "migraine," adding noise.
5. **No feature-based comparator:** without a Tier-1 classical baseline you cannot tell whether the failure is the model or the task. In fact, at n≈30 a classical feature model often beats deep learning.

---

## 10. Connecting the Detector to the Binaural-Beat System — Question 10

The existing control law (docs/ADAPTIVE_CONTROL_LAW_EXPLANATION.md) is a **state-space controller**: it reads normalized band-power proportions (δ, θ, α, β, γ), computes deviations from targets, and updates beat frequency accordingly. It already exists and is well-specified. What the detection pipeline adds is the **trait→state pathway**:

### Two distinct subsystems
```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. TRAIT CLASSIFIER (this pipeline)                                 │
│    Input:  30-s HD-EEG windows (or wearable subset in deployment)   │
│    Output: P(migraine disorder | EEG)                                │
│    Used for: screening, eligibility, personalization initialization │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  P(migraine), baseline deviation vector
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. STATE ESTIMATOR (real-time, per 30–60 s block)                   │
│    Input:  live band-power proportions, relative to THIS person's    │
│            baseline (from the initial/resting session)              │
│    Output: ΔP_k(t) — deviations from the individual target profile  │
│    Feeds:  the existing control law                                 │
│            f_b(t+Δt) = f_b(t) + α·Σ(w_k·ΔP_k) + β·E(t)              │
└─────────────────────────────────────┬────────────────────────────────┘
                                      ▼
                     Binaural beat generator (4–13 Hz)
```

### Concrete integration steps
1. **Personalization initialization (`f_b(0)`, `w_k(0)`):** use the trait classifier's subject-level deviation profile (band-power deficits/enhancements relative to the healthy controls' distribution) to seed the control law — the same logic the control law already uses (`f_b(0) = f_base + (P_α,target − P_α,baseline) × 5.0`; weights `w_k(0) = 0.5 + (P_k,target − P_k,baseline)`). The trait classifier effectively *provides the baseline* from a single short recording instead of requiring a long calibration session.
2. **Closed-loop monitoring:** after the intervention starts, the state estimator (not the trait classifier) drives the online updates. The trait classifier is re-run periodically as a safety check of "is the brain state still migraine-like / deviating from healthy norm."
3. **Outcome metric for the therapy study:** the controller's target band distribution (alpha 40%, theta 20%, beta 20%, delta 15%, gamma 5%) should be validated against the actual healthy-control band distribution estimated from this dataset — i.e., replace the *assumed* targets with *measured* targets from the control group. This transforms the control law from a heuristic into an empirically-calibrated controller.
4. **Trait→therapy personalization (science):** test whether the trait model's subtype (aura vs non-aura; alpha-deficit vs theta-excess profiles) predicts *which beat frequency* moves a given individual toward the healthy distribution (e.g., alpha-deficit patients benefit from alpha-range beats). This is a legitimate novel contribution (see Section 11).
5. **Deployment constraint:** the 128-channel model will not run on a wearable headband. Reserve a **channel-reduction study**: retrain/validate on a wearable-consistent subset (e.g., Fp1/Fp2/F7/F8/T7/T8/P7/P8/O1/O2 + central) and quantify the AUC drop. Expect a drop; report it honestly. The research claim stays on HD-EEG; the wearable is a follow-up engineering study.

---

## 11. Supported by Research vs. Novel Contributions — Question 11

| Claim / Component | Status |
|---|---|
| Interictal migraine alters EEG alpha, theta, and coherence | ✅ **Supported** (Bjørk 2009 review; Coppola 2009; Chamanzar 2020) |
| HD-EEG coherence is abnormal in this exact cohort | ✅ **Supported** (the dataset's own paper) |
| Subject-level CV, LOSO, leakage prevention, chance-level bounds are mandatory in small-n EEG ML | ✅ **Supported** (Varoquaux 2018; Varoquaux & Cheplygina 2022; Combrisson & Jerbi 2015) |
| Deep CNNs (EEGNet, Shallow/DeepConvNet) are standard EEG classifiers | ✅ **Supported** (Lawhern 2018; Schirrmeister 2017) |
| Self-supervised/contrastive EEG pretraining improves small-label performance | ✅ **Supported** (Banville 2021; Kostas 2021) |
| GAN augmentation can help EEG but is leak-prone and sample-hungry | ✅ **Supported as a caution** (Lashgari 2020 review; Hartmann 2018) — but see next row |
| Using a GAN on a 34-subject *single-site* dataset to enable DL | ⚠️ **Methodologically risky; needs justification & strict protocols** — paper will require editorial defense |
| **Binary migraine-vs-control classifier from the Chamanzar 128-ch dataset using PSD + connectivity + SSVEP/SSAEP fusion** | 🆕 **Novel contribution** — no published 128-ch subject-level classifier exists for this cohort |
| **3-condition fusion with subject-level LOSO benchmarking classical vs DL** | 🆕 **Novel contribution** (small but publishable as a methods/feasibility paper) |
| **Aura vs non-aura stratification in the classifier + subtype-driven beat-frequency prescription** | 🆕 **Novel contribution** |
| **Replacing assumed therapeutic band targets with measured healthy-control distributions from this dataset** | 🆕 **Novel contribution** (calibrates the existing control law) |
| Ictal ("currently in attack") migraine detection from this dataset | ❌ **Not supported by data; explicitly out of scope** |

---

## 12. Methodological Red Flags in the Current Implementation (action list)

1. **Resample the data or fix the fs** — the `TARGET_SFREQ` constant is dead code; the actual epochs are 1024 samples at native 512 Hz.
2. **Build labels from the demographics Excel**, not folder prefixes; run analyses with and without excluded subjects (M2, M6, M18, M13).
3. **Do not down-project 130→62 channels** for model training; design the DL arm for 130 channels (or a fixed 64-channel subset **selected a priori** for all subjects and conditions).
4. **Fix the prefix-matching bug** in any stacking/windowing code (`C1_` matches `C10_`).
5. **Add the `empty-epoch` guard** so failed recordings (C12) are recorded, not silently broken.
6. **Add a per-subject QC report** (channels kept, ICs rejected, % segments kept) and attach it to the paper appendix.
7. **Record condition and block structure** for SSAEP/SSVEP (4 vs 6 Hz) so stimulus-locked features and habituation slopes can be extracted. The current preprocessing destroys event/block info by using fixed-length epochs only.
8. **Never fit preprocessing scalers, ICA, or normalization on test folds.**

---

## 13. Realistic Expectations

- With ~14–17 healthy-clean migraine vs ~18 controls, subject-level AUC in the **0.70–0.85 range (with wide CIs)** would be a strong pilot result; AUC > 0.9 should trigger a leakage audit. AUC 0.5–0.6 (like the prior attempt) is consistent with (a) resting-only broadband features being weak, (b) channel-mismatch loss, and (c) n≈30 variance — all fixable.
- Publishable framing: **"Feasibility of interictal migraine trait detection from ultra-high-density EEG with subject-independent validation"** — a methods/feasibility contribution with negative results reported honestly.
- The binaural-beat integration is a **design/engineering contribution** at this stage; clinical efficacy (does the beat actually move this individual's EEG toward healthy band distributions) requires a separate IRB-approved intervention study with ictal/ambulatory data.

---

## 14. Key References

- Chamanzar, A., Haigh, S. M., Grover, P., & Behrmann, M. (2020). *Abnormalities in cortical pattern of coherence in migraine detected using ultra high-density EEG.* [Dataset paper — coherence abnormalities in this exact cohort]
- Bjørk, M. H., et al. (2009). *The quantitative EEG in migraine: a systematic review* — alpha slowing, theta increases, interictal effects.
- Coppola, G., Pierelli, F., & Schoenen, J. (2009). *Habituation and migraine.* Neurobiology of Learning and Memory — dyshabituation of evoked responses.
- Lawhern, V. J., et al. (2018). *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces.* J. Neural Eng.
- Schirrmeister, R. T., et al. (2017). *Deep learning with convolutional neural networks for EEG decoding and visualization.* Hum. Brain Mapp.
- Roy, Y., et al. (2019). *Deep learning-based electroencephalography analysis: a systematic review.* J. Neural Eng.
- Craik, A., He, Y., & Contreras-Vidal, J. L. (2019). *Deep learning for electroencephalogram (EEG) classification tasks.* J. Neural Eng.
- Varoquaux, G. (2018). *Cross-validation failure: small sample sizes lead to large error bars.* NeuroImage.
- Varoquaux, G., & Cheplygina, V. (2022). *Machine learning for medical imaging: methodological failures and recommendations for the future.* npj Digital Medicine.
- Combrisson, E., & Jerbi, K. (2015). *Exceeding chance level by chance: the caveat of theoretical chance levels in brain signal classification.* J. Neurosci. Methods.
- Pion-Tonachini, L., et al. (2019). *ICLabel (v1.2): a deep neural network toolbox.* NeuroImage.
- Mullen, T., et al. (2015). *Real-time neuroimaging and cognitive monitoring using wearable dry EEG* — ASR artifact subspace reconstruction.
- Bastos, A. M., & Schoffelen, J.-M. (2016). *A tutorial review of functional connectivity analysis methods and their interpretational pitfalls.* Front. Syst. Neurosci.
- Vinck, M., et al. (2011). *An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias.* NeuroImage — wPLI.
- Lashgari, E., Liang, D., & Maoz, U. (2020). *Data augmentation for deep-learning-based electroencephalography.* J. Neurosci. Methods.
- Hartmann, K. G., et al. (2018). *EEG-GAN: Generative adversarial networks for electroencephalograhic (EEG) brain signals.* arXiv.
- Banville, H., et al. (2021). *Self-supervised representation learning from electroencephalography signals.* Pattern Recognition.
- Kostas, D., Chakraborty, S., & Rudzicz, F. (2021). *BENDR: using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data.* Front. Hum. Neurosci.
- Garcia-Argibay, M., Santed, M. A., & Reales, J. M. (2019). *Efficacy of binaural auditory beats in cognition, anxiety, and pain perception: a meta- and systematic review.* Psychological Research.
- Cohen, M. X. (2014). *Analyzing Neural Time Series Data* — windowing/stationarity fundamentals.

---

*This document accompanies the FYDP-I project: Personalized Migraine Mitigation via Binaural Beats.*