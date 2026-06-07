# LEMON EEG Preprocessing Pipeline — Detailed Diagram & Flow

## 1. Overview: Generalized Pipeline Architecture

```
╔════════════════════════════════════════════════════════════════════════════╗
║                 LEMON EEG PREPROCESSING ARCHITECTURE                       ║
╚════════════════════════════════════════════════════════════════════════════╝

                          RAW BRAINVISION EEG
                   (128 channels, variable duration)
                                 │
                    ┌────────────┴────────────┐
                    │                         │
           ┌────────▼────────┐      ┌────────▼────────┐
           │  CHANNEL SETUP  │      │  METADATA LOAD  │
           │ (montage, ref)  │      │ (sampling rate) │
           └────────┬────────┘      └────────┬────────┘
                    │                         │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┴──────────────────────┐
          │      SIGNAL CONDITIONING STAGE              │
          │ (can run in parallel: filtering, resample)  │
          └──────────────┬───────────────────────────────┘
                         │
          ┌──────────────┼──────────────┬───────────┐
          │              │              │           │
     ┌────▼────┐    ┌────▼────┐   ┌────▼────┐  ┌──▼────┐
     │  Notch  │    │Bandpass │   │Resample │  │ Store │
     │ 50/100  │    │ 1-100Hz │   │ → 250Hz │  │ Stats │
     └────┬────┘    └────┬────┘   └────┬────┘  └──┬─────┘
          │              │              │          │
          └──────────────┼──────────────┴──────────┘
                         │
          ┌──────────────▼──────────────┐
          │  SPATIAL ARTIFACT REMOVAL   │
          │    (Bad Channel Path)       │
          └──────────────┬──────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼─────┐    ┌─────▼─────┐   ┌─────▼─────┐
   │PyPREP Bad│    │   Store   │   │  Temporal │
   │Channel   │    │ Bad Ch    │   │  Artifacts│
   │Detection │    │ Metadata  │   │   (ICA)   │
   └────┬─────┘    └─────┬─────┘   └─────┬─────┘
        │                │              │
   ┌────▼─────┐         │          ┌────▼─────┐
   │ Drop Bad │         │          │Re-ref &  │
   │Channels  │         │          │ Setup    │
   │(temporary)         │          └────┬─────┘
   └────┬─────┘         │              │
        │               │          ┌────▼─────┐
        └───────────────┼─────────→│   ICA    │
                        │          │Decompose │
                        │          └────┬─────┘
                        │              │
                        │          ┌────▼──────────┐
                        │          │ ICLabel       │
                        │          │ Classification│
                        │          └────┬──────────┘
                        │              │
                        │          ┌────▼────────┐
                        │          │Threshold    │
                        │          │ Rejection   │
                        │          │ IC.exclude  │
                        │          └────┬────────┘
                        │              │
                        │          ┌────▼────────────┐
                        │          │ ⚠️  QC1: IC      │
                        │          │ Reject Fraction │
                        │          └────┬────────────┘
                        │              │
                        │          ┌────▼────────┐
                        │          │Apply ICA    │
                        │          │(reconstruct)│
                        │          └────┬────────┘
                        │              │
        ┌───────────────┴──────────────┘
        │
        │         CHANNEL RESTORATION STAGE
        │       (Restore bad channels & interpolate)
        │
   ┌────▼────────────────┐
   │ Restore Bad         │
   │ Channels &          │
   │ Interpolate         │
   │ (spherical spline)  │
   └────┬────────────────┘
        │
   ┌────▼────────────────────────────┐
   │  TEMPORAL ARTIFACT ANNOTATION    │
   │    (Sliding Window Analysis)     │
   └────┬────────────────────────────┘
        │
   ┌────▼──────────────────────────────────────────┐
   │ Annotate Bad Segments                         │
   │ (sliding window: 0.5s, threshold: 250µV)      │
   └────┬──────────────────────────────────────────┘
        │
   ┌────▼────────────────────────────┐
   │  EPOCHING & SEGMENTATION        │
   │  (4s duration, 50% overlap)      │
   └────┬────────────────────────────┘
        │
   ┌────▼────────────────────────────┐
   │ Drop High-Amplitude Epochs      │
   │ (reject if max > 250µV)         │
   └────┬────────────────────────────┘
        │
   ┌────▼────────────────────────────┐
   │ ⚠️  QC2: Epoch Keep Ratio        │
   │ (warn if <50% retained)         │
   └────┬────────────────────────────┘
        │
   ┌────▼────────────────────────────┐
   │  SAVE OUTPUTS                   │
   │  .fif | .npy | metadata         │
   └────┬────────────────────────────┘
        │
        ▼
   ✅ READY FOR ML/ANALYSIS
```

---

## 2. Modular Processing Stages

```
╔════════════════════════════════════════════════════════════════════════════╗
║                    PREPROCESSING: PARALLEL VIEW                            ║
║           (Shows which steps can execute independently)                    ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INITIALIZATION (Sequential)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Load raw BrainVision file (.vhdr)                                        │
│  • Set channel types (EEG, EOG, reference)                                  │
│  • Apply 10-05 standard montage                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: SIGNAL CONDITIONING (Can parallelize notch/bandpass)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────┐    ┌─────────────┐    ┌──────────────┐               │
│   │  Notch Filter  │    │ Bandpass    │    │  Resample    │               │
│   │ 50 Hz, 100 Hz  │    │ 1-100 Hz    │    │ → 250 Hz     │               │
│   │                │    │             │    │              │               │
│   │ Time: ~5–10s   │    │ Time: ~5–8s │    │ Time: ~3–5s  │               │
│   └────────┬───────┘    └─────┬───────┘    └──────┬───────┘               │
│            └────────────┬──────┴─────────────────┘                         │
│                         ▼                                                   │
│                  (Filtered & resampled)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: SPATIAL QUALITY ASSESSMENT (PyPREP)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Run bad channel detection (deviation, NaN, correlation checks)            │
│ • Generates: list of bad channel indices                                    │
│ • Time: ~10–20s                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
           ┌────────▼────────┐   ┌────────▼────────┐
           │  Bad Channels?  │   │  Continue with  │
           │   Found         │   │  Clean Signal   │
           └────────┬────────┘   └────────┬────────┘
                    │                     │
        ┌───────────▼──────────┐         │
        │ Drop temporarily     │         │
        │ (for ICA stage)      │         │
        └───────────┬──────────┘         │
                    │                    │
                    └────────┬───────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: ARTIFACT DECOMPOSITION (ICA Path)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  4a. Re-reference (average ref)  →  Time: ~1–2s                             │
│      └─────────────────────────────────────┐                               │
│                                             ▼                               │
│  4b. ICA Fit (infomax, extended=True)  →  Time: ~30–60s (main bottleneck)  │
│      └─────────────────────────────────────┐                               │
│                                             ▼                               │
│  4c. ICLabel Classification (neural net)  Time: ~5–10s                     │
│      └─────────────────────────────────────┐                               │
│                                             ▼                               │
│  4d. Threshold-based IC Rejection  →  Time: <1s                            │
│      └─────────────────────────────────────┐                               │
│                                             ▼                               │
│  4e. Apply ICA (reconstruct signal)  →  Time: ~5–10s                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: CHANNEL RESTORATION & ANNOTATION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐      ┌──────────────────────┐                       │
│  │ Restore bad      │      │ Annotate bad         │                       │
│  │ channels         │      │ segments (sliding    │                       │
│  │ + interpolate    │      │ window, amplitude)   │                       │
│  │ Time: ~2–5s      │      │ Time: ~5–10s         │                       │
│  └────────┬─────────┘      └──────────┬───────────┘                       │
│           └──────────┬────────────────┘                                    │
│                      ▼                                                      │
│           (Full channel set restored)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: EPOCHING & FINAL QC (Segmentation)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  6a. Create epochs (4s, 50% overlap)  →  Time: ~2–3s                       │
│      └──────────────────────────────────┐                                  │
│                                          ▼                                  │
│  6b. Reject high-amplitude epochs  →  Time: ~1–2s                          │
│      └──────────────────────────────────┐                                  │
│                                          ▼                                  │
│  6c. Apply quality flags & metrics  →  Time: <1s                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: OUTPUT & PERSISTENCE                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Save .fif format (MNE-native)                                             │
│ • Save .npy format (NumPy for ML)                                           │
│ • Write metadata (CSV, channel names)                                       │
│ • Log to W&B (optional)                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                          ✅ COMPLETE
```

---

## 3. Decision Tree: Quality Gates & Conditional Paths

```
START (Raw EEG)
    │
    ├──→ [Filtering & Resampling] ──→ OK? ──→ YES ──────┐
    │                                 └──→ NO → LOG ERROR
    │
    ├──→ [Bad Channel Detection]
    │       │
    │       ├─→ n_bad_channels = 0-5
    │       │   Status: ✅ EXCELLENT
    │       │
    │       ├─→ n_bad_channels = 6-15
    │       │   Status: 🟡 ACCEPTABLE (but noisy)
    │       │
    │       └─→ n_bad_channels > 15
    │           Status: 🚩 FLAG (consider review)
    │           └──→ CONTINUE with caution
    │
    ├──→ [ICA Decomposition & Labeling]
    │       │
    │       ├─→ reject_ics_fraction = 0.05-0.15
    │       │   Status: ✅ CLEAN DATA
    │       │
    │       ├─→ reject_ics_fraction = 0.15-0.25
    │       │   Status: 🟡 MODERATE ARTIFACTS
    │       │   ⚠️  CHECKPOINT 1: Monitor
    │       │
    │       └─→ reject_ics_fraction ≥ 0.25
    │           Status: 🚩 FLAG (HIGH ARTIFACTS)
    │           ⚠️  CHECKPOINT 1: TRIGGERED
    │           Action: Log warning, continue
    │
    ├──→ [Apply ICA & Channel Restoration]
    │       │
    │       └──→ All steps successful? ──→ YES ─────┐
    │                                      └──→ NO → ERROR
    │
    ├──→ [Epoching]
    │       │
    │       ├─→ n_epochs_before ✓ computed
    │       │
    │       └─→ [Epoch Rejection]
    │               │
    │               ├─→ epoch_keep_ratio ≥ 0.80
    │               │   Status: ✅ EXCELLENT (lost <20%)
    │               │
    │               ├─→ epoch_keep_ratio = 0.50-0.80
    │               │   Status: 🟡 ACCEPTABLE (lost 20-50%)
    │               │   ⚠️  CHECKPOINT 2: Monitor
    │               │
    │               └─→ epoch_keep_ratio < 0.50
    │                   Status: 🚩 FLAG (POOR QUALITY)
    │                   ⚠️  CHECKPOINT 2: TRIGGERED
    │                   Action: Log warning, continue
    │
    └──→ [SAVE OUTPUTS]
            │
            └──→ ✅ COMPLETE
                 Results: .fif, .npy, metadata CSV
```

---

## 4. Detailed Step Breakdowns

---

## Detailed Step Breakdowns

### Bad Channel Detection ↔ IC Rejection Relationship

```
┌───────────────────────────────────────────────────────────────┐
│                    PREPROCESSING HIERARCHY                     │
└───────────────────────────────────────────────────────────────┘

SPATIAL ARTIFACTS (BAD CHANNELS)
────────────────────────────────
├─ Flat/dead electrode
├─ High impedance noise
├─ Environmental interference on one sensor
├─ Detection: Electrical properties (deviation, NaN, correlation)
└─ Action: REMOVE before ICA (improves decomposition)

                        ↓ (ICA works better without these)

TEMPORAL ARTIFACTS (ICA COMPONENTS)
───────────────────────────────────
├─ Eye blink (frontal, 100+ µV, ~500 ms duration)
├─ Muscle activity (jaw clench, neck tension, scalp muscles)
├─ Cardiac (synchronized with heartbeat, ~1 Hz)
├─ Line noise (exactly 50/100 Hz, coherent across channels)
└─ Detection: Statistical independence + ICLabel classifier
   Action: EXCLUDE components (zero out during reconstruction)


WHY BAD CHANNELS → ICA REJECTION RELATIONSHIP:
──────────────────────────────────────────────
1. Bad channels inject non-physiological variance
2. ICA tries to separate this noise as independent components
3. If bad channels are NOT removed first:
   → ICA wastes components capturing channel noise
   → Fewer components left for "real" artifacts (eyes, muscle)
   → Some artifacts may not be properly separated
4. Result: Higher rejection fraction, lower epoch quality

OPTIMAL ORDER:
──────────────
[Bad channels detected] → [Removed] → [ICA fit] → [Artifacts rejected]
       (spatial)                          (temporal)
```

### Quality Control Checkpoints

```
┌─────────────────────────────────────────────────────────────────┐
│              QUALITY FLAGS & DECISION POINTS                     │
└─────────────────────────────────────────────────────────────────┘

CHECKPOINT 1: n_bad_channels
─────────────────────────────
Result: count of channels marked as bad
Typical range: 0–15 channels out of 128
🟢 Good: 0–5 channels
🟡 Warning: 6–15 channels (noisy session)
🔴 Bad: >15 channels (severe noise; consider discarding)

CHECKPOINT 2: reject_ics_fraction
──────────────────────────────────
Result: (# rejected ICs) / (n_components)
Typical range: 0.05–0.30
Threshold: IC_REJECT_FRACTION_FLAG = 0.25

🟢 Good: <0.15 (few artifacts, clean data)
🟡 Warning: 0.15–0.25 (moderate artifact load)
🔴 Bad: >0.25 (high artifact contamination)

→ High rejection = subject moved a lot, noisy recording, or poor electrode contact

CHECKPOINT 3: n_epochs_before → n_epochs_after
─────────────────────────────────────────────
Result: epoch_keep_ratio = n_after / n_before
Typical range: 0.50–1.0
Threshold: EPOCH_KEEP_RATIO_FLAG = 0.50

🟢 Good: >0.80 (lost <20% of epochs)
🟡 Warning: 0.50–0.80 (lost 20–50% of epochs)
🔴 Bad: <0.50 (lost >50% of epochs)

→ Low keep ratio = remaining segments are noisy despite ICA and bad channel removal
```

---

## Processing Statistics Summary

```
PER-SUBJECT METRICS TRACKED:
───────────────────────────
├─ subject_id (unique identifier)
├─ status (success / failed)
├─ n_bad_channels (count)
├─ bad_channel_names (list)
├─ n_rejected_ics (count)
├─ reject_ics_fraction (ratio, 0–1)
├─ bad_segments_s (duration in seconds)
├─ n_epochs_before (count before rejection)
├─ n_epochs_after (count after rejection)
├─ epoch_keep_ratio (ratio, 0–1)
├─ epoch_shape (tuple: n_epochs × n_channels × n_times)
└─ elapsed_s (processing time)

COHORT-LEVEL SUMMARIES:
──────────────────────
├─ Success rate: (# successful) / (total subjects) %
├─ Mean bad channels per subject
├─ Mean rejected ICs per subject
├─ Mean epoch keep ratio
├─ Bad channel frequency (which channels failed most often)
└─ Processing time distribution
```

---

## Data Flow Diagram

```
SUBJECT-LEVEL PIPELINE:

    Raw BrainVision
         File
          ↓
    [Read & Load]
          ↓
    [Setup Channels & Montage]
          ↓
    [Notch + Bandpass + Resample]
          ↓
    [Bad Channel Detection] ─→ Store bad_channel_names
          ↓
    [Drop Bad Channels]
          ↓
    [Average Re-reference]
          ↓
    [ICA Decomposition]
          ↓
    [ICLabel Probabilities]
          ↓
    [Threshold-based IC Rejection] ─→ Store n_rejected_ics
          ↓
    [Apply ICA (exclude comps)]
          ↓
    [Restore & Interpolate Bad Channels]
          ↓
    [Annotate Bad Segments (sliding window)]
          ↓
    [Create Fixed-Length Epochs (4s, 50% overlap)]
          ↓
    [Drop High-Amplitude Epochs]
          ↓
    [Save .fif + .npy + metadata]
          ↓
    ✅ Processed Subject


BATCH-LEVEL PIPELINE:

    [Load all subject directories]
          ↓
    For each subject:
    ├─ Check if already processed (skip if yes)
    ├─ Check if .vhdr file exists
    ├─ Call subject-level pipeline
    ├─ Collect metrics & timings
    ├─ Log to W&B (optional)
    └─ Continue to next subject
          ↓
    [Aggregate results into DataFrame]
          ↓
    [Export preprocessing_metrics_per_subject.csv]
    [Export bad_channel_frequency.csv]
          ↓
    [Generate publication-quality plots]
    ├─ Per-subject success outcomes
    ├─ Bad channels per subject
    ├─ ICA rejection fraction distribution
    ├─ Epoch keep ratio by status
    ├─ Bad channel heatmap
    ├─ Runtime distribution
    └─ Summary diagnostics grid
          ↓
    ✅ Batch Complete
```

---

## Parameter Tuning Guide

```
KEY PARAMETERS & SENSITIVITY:

1. IC_REJECTION_THRESHOLD (default: 0.80)
   ────────────────────────────────────
   • Higher (0.90):  Conservative; keep more ICs (may leave artifacts)
   • Lower (0.70):   Aggressive; reject more ICs (may remove brain signal)
   → Tune based on visual inspection of ICA component maps

2. AMPLITUDE_REJECT_UV (default: 250 µV)
   ─────────────────────────────────────
   • Higher (300 µV): Keep more epochs (loose quality)
   • Lower (200 µV): Stricter quality (lose more data)
   → Adjust based on typical noise levels in your setup

3. EPOCH_DURATION (default: 4.0 s) & EPOCH_OVERLAP (default: 0.5)
   ──────────────────────────────────────────────────────────────
   • Longer epochs: More context, fewer samples
   • More overlap: More training examples, higher redundancy
   → Chosen to match downstream model requirements

4. TARGET_SFREQ (default: 250 Hz)
   ──────────────────────────────
   • Higher (500 Hz): Preserve fast dynamics, larger files
   • Lower (125 Hz): Reduce file size, may lose detail
   → Set to match other datasets for consistency

5. IC_REJECT_FRACTION_FLAG & EPOCH_KEEP_RATIO_FLAG
   ───────────────────────────────────────────────
   • These are warning thresholds, not hard stops
   • Use for flagging problematic subjects for manual review
   • Can still process; flag for downstream quality checks
```

---

## Example Output

```
SUCCESSFUL PREPROCESSING EXAMPLE:

Subject: sub-010002
────────────────────
✅ n_bad_channels: 3 (Cz, POz, Oz)
✅ reject_ics_fraction: 0.12 (12% of ~120 components)
✅ n_epochs_before: 287
✅ n_epochs_after: 281
✅ epoch_keep_ratio: 0.98 (98% retained)
✅ bad_segments_s: 1.5 s out of 1148 s total
✅ elapsed_s: 45.3 s

Output files:
├─ sub-010002-epo.fif (128 channels, 281 epochs, 4s each)
├─ sub-010002_epochs.npy (281, 128, 1000)
└─ metadata in CSV


FLAGGED PREPROCESSING EXAMPLE:

Subject: sub-010015
──────────────────
⚠️  n_bad_channels: 12 (electrode contact issues)
⚠️  reject_ics_fraction: 0.28 (exceeds FLAG of 0.25)
⚠️  n_epochs_before: 287
⚠️  n_epochs_after: 152
⚠️  epoch_keep_ratio: 0.53 (borderline; FLAG = 0.50)
⚠️  bad_segments_s: 18.2 s (noisy recording)
✓  elapsed_s: 42.1 s

→ Recommendation: Review manually; consider reprocessing with adjusted parameters
   or exclude from analysis if noise is too severe.
```

---

## Visual Layout for Research Methodology

Use this structure for figures in your thesis/paper:

**Figure 1: High-Level Overview**
- Raw EEG → Filtering → Channel QC → ICA → Epoching → Clean Data

**Figure 2: Bad Channel Detection Flow**
- Input raw signal → Spatial statistics → Identify noisy channels

**Figure 3: ICA Artifact Rejection Flow**
- Clean signal → Decompose → Label components → Reject artifacts → Reconstruct

**Figure 4: Quality Control Gates**
- Decision tree showing how metrics determine subject inclusion/exclusion

**Figure 5: Data Shape Transformations**
- Show shape changes at each step (raw → epochs → ML-ready)

**Figure 6: Temporal Segmentation**
- Illustration of sliding window, bad segment annotation, and epoching

---

## Copy-Paste Snippets for Papers

**Methods Section:**
> "Raw EEG was preprocessed using MNE-Python following a standardized pipeline:
> (1) notch filtering at 50 and 100 Hz; (2) bandpass filtering (1–100 Hz);
> (3) resampling to 250 Hz; (4) bad channel detection using PyPREP;
> (5) average re-referencing; (6) Independent Component Analysis with
> ICLabel-based artifact rejection (threshold: 0.80 probability);
> (7) interpolation of removed channels; (8) amplitude-based bad segment
> annotation; (9) fixed-length epoching (4 s, 50% overlap);
> (10) epoch rejection (>250 µV). Quality flags were applied to identify
> problematic subjects: IC rejection fraction >0.25 or epoch keep ratio <0.50."

