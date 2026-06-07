# LEMON Full Batch Preprocessing Documentation

## 1. What This Notebook Does

The notebook [LEMON_Full_Batch_Only.ipynb](../LEMON_Full_Batch_Only.ipynb) is a full batch preprocessing workflow for the LEMON EEG dataset. Its purpose is to take raw BrainVision recordings from many subjects and turn them into cleaner, standardized EEG epochs that can be used later for machine learning, analysis, and model training.

The notebook is intentionally batch-only. That means it does not focus on one example subject for demonstration. Instead, it is designed to process the whole dataset in one consistent pipeline so that every subject is treated in the same way. This is important because a uniform preprocessing strategy makes the results easier to compare and more suitable for downstream modeling.

## 2. Why This Workflow Is Structured This Way

EEG data is sensitive to noise, bad sensors, movement, eye activity, and recording differences across participants. If these issues are not handled early, later analysis can become misleading or unstable. The pipeline therefore follows a careful order:

1. Clean the raw signal.
2. Identify noisy channels.
3. Remove or reduce artifact-related components.
4. Mark visibly corrupted time segments.
5. Split the signal into standardized epochs.
6. Save the result in reusable file formats.

This structure is practical because each step makes the next step more reliable. For example, removing bad channels before ICA improves the quality of the component separation. Likewise, epoching only after cleaning helps ensure that the final dataset is more consistent and easier to reuse.

## 3. Data Flow Overview

### Input

The notebook reads raw EEG recordings from the LEMON folder structure, mainly from BrainVision files:

- `.vhdr` header files
- `.eeg` signal files
- `.vmrk` marker files

### Output

Cleaned outputs are saved in `data/LEMON_preprocessed` as:

- `.fif` epoch files for MNE compatibility
- `.npy` arrays for machine learning workflows
- `channel_names.txt` for consistent channel reference
- CSV summaries for tracking results across subjects
- figures in a `figures` subfolder for reporting and review

## 4. Main Libraries Used

| Library | Role in the notebook | Why it matters |
| --- | --- | --- |
| `mne` | EEG loading, filtering, re-referencing, ICA, epoching, saving | This is the core EEG analysis library and handles the scientific preprocessing steps |
| `numpy` | Numeric operations and array handling | Useful for signal calculations and storing epoch data |
| `pandas` | Tabular summaries and CSV export | Makes it easy to organize subject results and report outcomes |
| `mne_icalabel` | Automatic ICA component labeling | Helps identify components linked to eye, muscle, or other artifacts |
| `pyprep` | Bad-channel detection | Provides a robust way to detect unreliable EEG channels |
| `cupy` | Optional GPU acceleration | Speeds up some calculations when a CUDA-capable system is available |
| `wandb` | Optional tracking and logging | Useful for monitoring the batch run and keeping a processing record |
| `matplotlib` / `seaborn` | Plotting and visualization | Used to summarize outcomes in a readable visual form |

The notebook is built so that it still runs even if some optional libraries are missing. In particular, `wandb` and `cupy` are treated as optional enhancements rather than strict requirements.

## 5. Parameters and What They Mean

| Parameter | Value | Meaning | Why it is used |
| --- | --- | --- | --- |
| `TARGET_SFREQ` | `250` | Final sampling rate | A balanced rate that keeps EEG detail while reducing file size and computation time |
| `LINE_NOISE_FREQ` | `50` | Power-line noise frequency | Removes electrical interference common in many recording environments |
| `HIGHPASS_FREQ` | `1.0` | High-pass filter cutoff | Reduces slow drift and baseline wandering |
| `LOWPASS_FREQ` | `100.0` | Low-pass filter cutoff | Keeps useful EEG information while reducing high-frequency noise |
| `IC_REJECTION_THRESHOLD` | `0.80` | Confidence threshold for rejecting ICA components | Keeps rejection conservative so only likely artifact components are removed |
| `EPOCH_DURATION` | `4.0` | Length of each epoch in seconds | Produces segments long enough to contain useful EEG context |
| `EPOCH_OVERLAP` | `0.5` | 50% overlap between epochs | Increases the number of usable samples without losing continuity |
| `AMPLITUDE_REJECT_UV` | `250e-6` | Peak-to-peak rejection threshold | Removes epochs that are too noisy or too large in amplitude |
| `IC_REJECT_FRACTION_FLAG` | `0.25` | Quality flag threshold for ICA rejection rate | Warns when too many components are removed |
| `EPOCH_KEEP_RATIO_FLAG` | `0.5` | Quality flag threshold for epoch retention | Warns when too much data is lost during epoch rejection |

These parameters reflect a practical balance between signal quality and data retention. The notebook is not trying to over-clean the EEG to the point of losing data; instead, it aims to remove the most obvious noise while keeping enough usable signal for later analysis.

## 6. Step-by-Step Pipeline

### 6.1 Environment Checks and Imports

The notebook first imports the required libraries and checks whether optional tools are available. It also tries to detect a CUDA installation so that GPU acceleration can be enabled when possible.

This is done because EEG preprocessing can be computationally expensive, especially when the batch includes many subjects. Using GPU support when available can reduce runtime, but the notebook still needs to work on standard CPU-only machines.

### 6.2 Optional Weights and Biases Logging

If `wandb` is installed and configured, the notebook opens a logging session and records subject-level progress, summary metrics, and a final table of results.

This is useful because long preprocessing jobs can take a lot of time. Logging makes it easier to monitor progress, identify failures, and review the run later without manually checking every subject.

### 6.3 Bad Channel Detection

The helper function `detect_bad_channels` uses PyPREP to identify channels that appear unreliable. It checks for several common problems such as flat signals, abnormal deviation, excessive high-frequency noise, and poor correlation with neighboring channels. It also attempts a RANSAC-based check when possible.

This step is important because a noisy channel can distort the rest of the preprocessing pipeline. Identifying bad channels early helps protect later steps such as filtering, re-referencing, and ICA.

### 6.4 Bad Segment Annotation

The helper function `annotate_bad_segments` scans the EEG in short windows and marks windows as bad when the peak-to-peak amplitude becomes too large. In this notebook, the window size is 0.5 seconds and the threshold is 250 microvolts.

The reason for this approach is that some artifacts are brief but strong, such as movement or sudden electrical interference. Rather than removing the entire recording, the notebook marks only the affected segments so that the rest of the data can still be used.

### 6.5 Subject-Level Preprocessing Function

The heart of the notebook is the function `preprocess_lemon_subject`. It is called once for each subject and performs the full cleaning pipeline.

#### a. Load the Raw EEG

The BrainVision file is read into memory with `preload=True`, which allows the signal to be processed efficiently.

#### b. Prepare the Channels

The notebook marks `VEOG` as an eye-related channel, adds `FCz` if it is missing, and applies the standard 10-05 montage.

This matters because EEG processing tools work better when the channel layout is known. A montage gives the signal a spatial structure, which is needed for channel-level interpretation and interpolation.

#### c. Filter and Resample

The signal is notched at 50 Hz and its harmonic, high-pass filtered at 1 Hz, low-pass filtered at 100 Hz, and then resampled to 250 Hz.

This is done to clean up the signal and make all subjects share the same sampling rate. A consistent sampling rate is especially useful when the outputs will later be used in machine learning models.

#### d. Detect and Temporarily Remove Bad Channels

The detected bad channels are stored and temporarily removed before ICA.

This step improves ICA quality because artifact-heavy or broken channels can otherwise affect the decomposition.

#### e. Re-reference the Signal

The cleaned signal is re-referenced using the average reference.

Average referencing is a common EEG strategy because it reduces the influence of any one electrode and gives a more balanced view of brain activity.

#### f. Run ICA and Label Components

Independent Component Analysis is performed using the `infomax` method with `extended=True`. The resulting components are then labeled with ICLabel.

The notebook rejects components that are confidently identified as artifacts such as eye activity, muscle activity, heart beat, line noise, channel noise, or other non-brain sources.

This is one of the most important cleaning steps because it removes structured artifact patterns while preserving the underlying EEG signal as much as possible.

#### g. Restore Removed Channels

Any bad channels that were removed are added back, marked as bad, and interpolated.

This makes the final signal more complete again. Instead of permanently deleting channels, the notebook estimates their likely values based on surrounding information, which helps keep the data shape consistent across subjects.

#### h. Annotate Remaining Bad Segments

The notebook marks high-amplitude segments as artifacts so they can be excluded from the epoching stage.

#### i. Create Fixed-Length Epochs

The continuous EEG is converted into 4-second overlapping epochs with 50% overlap.

This is helpful because fixed-length epochs are easier to compare across subjects and easier to feed into downstream models. Overlap also improves data utilization by creating more training examples from the same recording.

#### j. Reject Noisy Epochs

Epochs that exceed the amplitude threshold are removed.

This final cleaning step helps make the exported data more reliable by excluding segments that still show obvious noise after earlier processing.

#### k. Save the Results

Each subject produces a saved `.fif` file and `.npy` file. The notebook also writes `channel_names.txt` once, so the saved data can be interpreted consistently later.

### 6.6 Batch Processing Loop

The notebook loops through all subject folders in the dataset and handles each case in a practical way:

1. If the subject has already been processed, it is skipped.
2. If the `.vhdr` file is missing, the subject is recorded as missing.
3. If the subject is ready, the preprocessing function is executed.
4. Processing time and counts are tracked for later review.

This design makes the notebook safe to rerun. If the batch is interrupted, already completed subjects do not need to be repeated from scratch.

### 6.7 Final Summary and Reporting

After all subjects are processed, the notebook builds a table of results and saves a CSV summary. It also creates plots showing:

- subject-wise success or failure
- number of bad channels per subject
- ICA rejection fraction
- epoch retention rate
- bad-channel frequency across the cohort
- a bad-channel heatmap
- runtime diagnostics

These plots are useful because they turn the preprocessing run into something that can be interpreted at a glance. Instead of only storing cleaned data, the notebook also records how clean each subject was and where the main problems appeared.

## 7. Why the Final Outputs Are Saved in Two Formats

The notebook saves both `.fif` and `.npy` outputs because they serve different purposes:

- `.fif` is the native MNE format and is useful if the data will be revisited with EEG tools later.
- `.npy` is simple and efficient for machine learning workflows.

Keeping both formats makes the pipeline more flexible and reduces the need to preprocess the data again later.

## 8. What the Quality Flags Mean

The notebook does not only clean the data; it also checks whether the cleaning became too aggressive.

- A high ICA rejection fraction suggests that the subject may have had strong artifact contamination.
- A low epoch keep ratio suggests that a large portion of the recording was discarded.

These are warnings rather than hard failures. The purpose is to help the researcher review subjects that may need special attention without stopping the whole batch.

## 9. Practical Value of the Notebook

This preprocessing notebook is useful because it creates a reproducible and consistent dataset from raw EEG recordings. That consistency is especially important in projects like this one, where the preprocessed EEG may later support:

- feature extraction
- group comparisons
- model training
- transfer learning
- clinical or intervention-oriented analysis

In simple terms, the notebook turns noisy raw EEG into a cleaner research dataset that is easier to trust and reuse.

## 10. Short Summary

In one sentence, this notebook takes raw LEMON EEG recordings, cleans them using standard EEG methods, segments them into reusable epochs, and saves the results in analysis-friendly formats with summary statistics and plots.

The overall design is sensible because it combines signal cleaning, quality control, automation, and reporting in one reproducible batch workflow.
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
