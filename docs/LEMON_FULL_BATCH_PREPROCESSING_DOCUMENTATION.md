# LEMON Full Batch Preprocessing Documentation

## Project Context

This document describes the full preprocessing workflow implemented in:

- `LEMON_Full_Batch_Only.ipynb`

The notebook performs automated, subject-wise EEG preprocessing for all LEMON participants in batch mode, with optional Weights and Biases (W&B) tracking.

## Followed Reference Paper

The preprocessing workflow is based on the pipeline referenced in the project as:

- **DISCOVER-EEG (Gil Avila et al., 2023, Scientific Data)**

In your repository, this is stated explicitly in `LEMON_Preprocessing_Pipeline.ipynb` as the guiding paper and pipeline source.

## Objective of This Pipeline

The pipeline transforms raw BrainVision EEG recordings into clean, fixed-format epochs suitable for downstream ML tasks (transfer learning and migraine classifier fine-tuning).

Primary goals:

1. Remove line noise, drift, and artifacts.
2. Detect and repair noisy channels.
3. Remove ICA components likely representing eye and muscle artifacts.
4. Remove high-amplitude bad time segments.
5. Create overlapping fixed-length epochs with consistent shape.
6. Save outputs in reproducible formats (`.fif` and `.npy`).

## Data Inputs and Outputs

### Input

- **Raw source folder:**
  - `EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/sub-*/RSEEG/sub-*.vhdr`
- **File format:** BrainVision (`.vhdr`, `.eeg`, `.vmrk`)

### Output

- **Output folder:**
  - `data/LEMON_preprocessed`
- **Per subject output files:**
  - `sub-XXXXXX-epo.fif`
  - `sub-XXXXXX_epochs.npy`
- **Shared output files:**
  - `channel_names.txt` (created once)
  - `preprocessing_summary.csv`

## High-Level Pipeline Stages

1. Environment setup and import checks.
2. Parameter initialization.
3. Bad-channel detection helper.
4. Bad-segment annotation helper.
5. Subject-level preprocessing function.
6. Batch loop over all subjects.
7. Final summary and CSV export.

## Detailed Step-by-Step Workflow

### Step 1: Imports and Runtime Checks

Libraries used:

- `mne`, `numpy`, `pandas`, `glob`, `os`, `time`, `warnings`
- `mne_icalabel` for automatic ICA component labeling
- Optional: `wandb` for experiment logging
- Optional: `cupy` + `mne.cuda` for GPU acceleration

Behavior:

- Tries to enable CUDA filtering/resampling if available.
- Falls back to CPU mode if CUDA/CuPy is unavailable.

### Step 2: Paths and Global Parameters

Configured parameters in `LEMON_Full_Batch_Only.ipynb`:

- `TARGET_SFREQ = 250`
- `LINE_NOISE_FREQ = 50`
- `HIGHPASS_FREQ = 1.0`
- `IC_REJECTION_THRESHOLD = 0.80`
- `EPOCH_DURATION = 4.0`
- `EPOCH_OVERLAP = 0.5`
- `AMPLITUDE_REJECT_UV = 250e-6`

Notes:

- The pipeline is intentionally set to 250 Hz to align with downstream data integration and model expectations.
- Epoch size here is 4 seconds with 50% overlap (different from the separate 2-second demonstration notebook).

### Step 2b: Optional W&B Tracking

If `wandb` is installed and configured:

- Initializes run metadata.
- Logs subject-level progress and running success rate.
- Logs final summary table and aggregate metrics.

If unavailable or fails:

- Continues preprocessing without interrupting the pipeline.

### Step 3: Bad Channel Detection (`detect_bad_channels`)

Method:

- Uses PyPREP `NoisyChannels` logic (clean_rawdata-style behavior).
- Evaluates channels using:
  - NaN/flat criteria
  - deviation
  - high-frequency noise
  - correlation
- Attempts RANSAC check when possible.

Implementation details:

- Excludes `FCz` during this detection stage (reference handling).
- Returns:
  - sorted bad channel list
  - whether RANSAC was successfully used

### Step 4: Bad Segment Annotation (`annotate_bad_segments`)

Method:

- Sliding window peak-to-peak amplitude check.
- Converts EEG to microvolts.
- Marks windows as `BAD_artifact` if any channel exceeds threshold.

Configured call in full batch function:

- threshold: 250 uV
- window: 0.5 s

Output:

- Total bad-segment duration in seconds (`bad_segments_s`).

### Step 5: Subject-Level Preprocessing (`preprocess_lemon_subject`)

This is the core function, executed once per subject.

#### 5.1 Load Raw EEG

- Reads BrainVision file with preload enabled.

#### 5.2 Channel Setup

- Marks `VEOG` as EOG.
- Adds `FCz` if missing.
- Applies `standard_1005` montage.

#### 5.3 Filtering and Resampling

- Notch filter at `50` and `100` Hz (power line and harmonic).
- High-pass filter at `1.0` Hz.
- Resample to `250` Hz.

#### 5.4 Bad Channel Marking and Temporary Removal

- Detects bad channels via PyPREP helper.
- Stores count in result dictionary.
- Drops bad channels before ICA.

#### 5.5 Re-reference

- Applies average reference (`projection=False`).

#### 5.6 ICA + ICLabel Artifact Rejection

- ICA method: `infomax`, `extended=True`.
- Number of components: `n_eeg - 1`.
- ICLabel probabilities are computed.
- Rejects ICs when:
  - `P(muscle) > 0.80` OR
  - `P(eye) > 0.80`

#### 5.7 Reconstruct Dropped Channels

- Adds removed channels back.
- Marks them as bad.
- Restores montage.
- Runs spherical interpolation (`interpolate_bads`).

#### 5.8 Bad Time Segment Annotation

- Adds `BAD_artifact` annotations based on amplitude thresholding.

#### 5.9 Epoching

- Keeps EEG channels only.
- Uses fixed-length epochs:
  - duration = 4.0 s
  - overlap = 2.0 s (50%)

#### 5.10 Epoch Rejection

- Drops epochs with EEG peak-to-peak > `250e-6` V.
- Stores counts before and after rejection.

#### 5.11 Save Outputs

- Saves `.fif` epochs file.
- Saves `.npy` array of epochs data.
- Saves `channel_names.txt` if not already present.

#### 5.12 Structured Result Return

Returns per-subject dictionary including:

- status (`success`/`failed`)
- elapsed time
- bad-channel count
- rejected-IC count
- bad segment duration
- epochs before/after
- epoch tensor shape
- error text when failed

### Step 6: Full Batch Loop

Batch logic in notebook:

1. Enumerates all subject directories.
2. Skips subjects already processed (`*_epochs.npy` exists).
3. Skips subjects with missing `.vhdr`.
4. Processes remaining subjects with timing.
5. Maintains counters:
   - `success_count`
   - `failed_count`
   - `missing_count`
   - `skipped_count`
   - `active_attempts`
6. Optionally logs each step to W&B.

### Step 7: Summary and Export

After batch completion:

- Builds DataFrame from successful subjects.
- Prints aggregate statistics.
- Saves CSV summary:
  - `data/LEMON_preprocessed/preprocessing_summary.csv`
- If W&B active, logs final table and summary metrics, then closes run.

## Output Data Contract for Downstream ML

Typical structure:

- shape: `(n_epochs, n_channels, n_times)`
- channels: EEG-only after cleaning and channel restoration
- sample rate: 250 Hz
- epoch duration: 4 s

This stable contract is crucial for:

- LEMON encoder pretraining compatibility.
- Migraine classifier fine-tuning pipeline.
- Reproducibility across subjects.

## Differences vs Paper-Level Baselines

Because this is an MNE/Python implementation and a project-specific batch pipeline, there may be practical adaptations from canonical EEGLAB/MATLAB implementations, including:

- MNE-equivalent filtering and epoching APIs.
- PyPREP-based bad-channel detection fallback behavior.
- Practical exception handling for robust full-batch execution.
- Batch skip logic for incremental reruns.
- Integrated experiment logging (W&B), not required by baseline paper workflows.

## Reproducibility Notes

1. `RANDOM_STATE = 42` is used for deterministic ICA setup.
2. GPU mode is optional and auto-detected.
3. The function captures failures per subject instead of stopping the whole batch.
4. Subject outputs are saved independently, enabling resumable preprocessing.

## Practical Run Checklist

Before running full batch:

1. Confirm dataset paths are correct.
2. Ensure `mne`, `mne_icalabel`, and `pyprep` are installed.
3. Optionally install/configure `wandb`.
4. Verify enough disk space for all `.fif` and `.npy` files.

After running full batch:

1. Inspect `preprocessing_summary.csv`.
2. Spot-check a few subjects by loading saved epoch files.
3. Verify channel consistency using `channel_names.txt`.
4. Confirm expected success/skip/failure counts.

## Citation Statement for This Documentation

This preprocessing workflow follows the project's referenced pipeline basis:

- **DISCOVER-EEG (Gil Avila et al., 2023, Scientific Data)**

as explicitly documented in `LEMON_Preprocessing_Pipeline.ipynb`, and adapted into a robust full-batch MNE implementation in `LEMON_Full_Batch_Only.ipynb`.
