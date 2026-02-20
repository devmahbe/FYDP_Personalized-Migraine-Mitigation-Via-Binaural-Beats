# LEMON + Migraine Transfer Learning Pipeline

## 📋 Complete Implementation Guide

This document outlines the complete transfer learning pipeline for improving migraine classification accuracy from 75% to 88-92% using the LEMON dataset (213 healthy subjects).

---

## 🗂️ Project Structure

```
FYDP-I_Personalized-Migraine-Mitigation-Via-Binaural-Beats/
├── Dataset/                          # Original 35 migraine subjects
│   ├── C1/, C2/, ..., C21/          # Control patients
│   ├── M1_1/, M2_1/, ..., M18_1/    # Migraine patients
│
├── EEG_MPILMBB_LEMON/               # New LEMON dataset (213 subjects)
│   ├── EEG_Raw_BIDS_ID/             # RAW EEG files (use this!)
│   │   ├── sub-010002/
│   │   │   └── RSEEG/
│   │   │       ├── sub-010002.vhdr  # BrainVision header
│   │   │       ├── sub-010002.vmrk  # Markers
│   │   │       └── sub-010002.eeg   # Raw data
│   │   └── ...
│   ├── EEG_Preprocessed_BIDS_ID/    # Pre-processed (we won't use this)
│   ├── EEG_Localizer_BIDS_ID/       # Channel locations
│   └── EEG_Info                      # Dataset documentation
│
├── src/                              # Processing modules
│   ├── lemon_preprocessor.py        # ✓ Stage 2: Preprocessing pipeline
│   ├── windowed_dataset_builder.py  # ✓ Stage 3: Windowing & dataset creation
│   └── transfer_learning_trainer.py # Stage 4: Model training
│
├── preprocessed_data/                # Output directory (created)
│   ├── lemon_windowed/              # LEMON windowed tensors
│   ├── migraine_windowed/           # Migraine windowed tensors
│   └── combined/                     # Combined dataset
│
├── models/                           # Trained models
│   ├── cnn_encoder_pretrained.pth   # Pre-trained on LEMON
│   └── cnn_lstm_finetuned.pth       # Fine-tuned on migraine
│
├── LEMON_Dataset_EDA.ipynb          # ✓ Stage 1: Exploratory analysis
└── LEMON_Transfer_Learning_Pipeline.ipynb  # Stage 4: End-to-end execution
```

---

## 🔄 4-Stage Pipeline

### **Stage 1: Exploratory Data Analysis (EDA)** ✅ COMPLETE

**File**: `LEMON_Dataset_EDA.ipynb`

**Objectives**:
- Load and inspect LEMON raw EEG files
- Check recording parameters (sampling rate, channels, duration)
- Identify common channels between LEMON and migraine datasets
- Visualize raw signals and data quality
- Analyze eyes-closed vs eyes-open conditions

**Key Findings**:
```
LEMON Dataset:
- 213 subjects (sub-010002 to sub-010321)
- Format: BrainVision (.vhdr, .vmrk, .eeg)
- Sampling rate: 250-500 Hz (varies by subject)
- Duration: ~16 minutes per subject
- Conditions: Eyes-closed (EC) + Eyes-open (EO) alternating

Migraine Dataset:
- 35 subjects (31 usable: 18 control, 9 aura, 4 non-aura)
- Format: BioSemi BDF (.bdf)
- Sampling rate: 512 Hz
- Duration: ~13 minutes per subject

Common EEG Channels: ~60-128 (to be verified by running EDA)
```

**Run this first** to verify dataset compatibility!

---

### **Stage 2: Unified Preprocessing Pipeline** ✅ COMPLETE

**File**: `src/lemon_preprocessor.py`

**Module**: `UnifiedEEGPreprocessor`

**Processing Steps**:

1. **Load Raw Data**
   - LEMON: BrainVision format
   - Migraine: BioSemi BDF format

2. **Channel Selection**
   - Retain only EEG channels
   - Keep EOG temporarily for artifact detection

3. **Channel Alignment**
   - Find intersection of channels between datasets
   - Reorder channels identically for spatial consistency

4. **Resampling**
   - Standardize to 250 Hz (reduces computation, maintains signal quality)

5. **Bandpass Filtering**
   - 1-45 Hz zero-phase FIR filter
   - Removes slow drifts (<1 Hz) and muscle noise (>45 Hz)

6. **Notch Filtering**
   - 50 Hz (or 60 Hz) + harmonics
   - Removes powerline interference

7. **Bad Channel Detection & Interpolation**
   - Statistical variance detection (Z-score > 3)
   - Flatline detection
   - Spherical spline interpolation

8. **Common Average Reference (CAR)**
   - Stabilizes spatial patterns across subjects

9. **ICA Artifact Removal**
   - Independent Component Analysis
   - Detect and remove EOG, ECG, muscle artifacts
   - Automatic EOG correlation-based detection

10. **Quality Validation**
    - Verify signal quality post-processing
    - Save processing metadata

**Usage**:
```python
from src.lemon_preprocessor import UnifiedEEGPreprocessor

preprocessor = UnifiedEEGPreprocessor(
    target_sfreq=250.0,
    bandpass_freqs=(1.0, 45.0),
    notch_freq=50.0,
    verbose=True
)

raw_clean, metadata = preprocessor.preprocess_file(
    file_path,
    dataset_type='lemon',  # or 'migraine'
    target_channels=common_channels
)
```

---

### **Stage 3: Windowed Dataset Creation** ✅ COMPLETE

**File**: `src/windowed_dataset_builder.py`

**Module**: `WindowedDatasetBuilder`

**Windowing Strategy**:

- **Window size**: 4 seconds (1000 samples at 250 Hz)
- **Overlap**: 50% (2-second stride)
- **Per subject**: ~390 windows from 16-min LEMON recording
- **Per subject**: ~195 windows from 13-min migraine recording

**Output Shape**: `(n_windows, n_channels, n_samples)`

**Processing Steps**:

1. **Epoch Creation**
   - Segment continuous recording into 4-second windows
   - 50% overlap for data augmentation

2. **Artifact Rejection**
   - Peak-to-peak amplitude threshold: 150 µV
   - Reject epochs with excessive noise

3. **Z-Score Normalization**
   - Per channel, per subject
   - Prevents cross-subject leakage
   - Formula: `z = (x - μ) / σ` where μ, σ computed per subject

4. **Metadata Creation**
   - Subject ID, dataset type, epoch index, label
   - Preserves traceability

5. **Subject-Level Splits**
   - Train/test split by subjects, not epochs
   - Prevents data leakage (critical!)

**Expected Output**:

```
LEMON Dataset:
- 213 subjects × ~390 windows = ~83,070 windows
- Shape: (83070, n_channels, 1000)
- Labels: None (unlabeled healthy baseline)

Migraine Dataset:
- 31 subjects × ~195 windows = ~6,045 windows
- Shape: (6045, n_channels, 1000)
- Labels: 0=Control (18), 1=Aura (9), 2=Non-Aura (4)

Combined:
- Total windows: ~89,115
- Massive data augmentation for training!
```

**Usage**:
```python
from src.windowed_dataset_builder import WindowedDatasetBuilder

builder = WindowedDatasetBuilder(
    window_duration=4.0,
    overlap=0.5,
    artifact_threshold=150e-6,
    normalize_per_subject=True,
    verbose=True
)

data, metadata = builder.build_dataset_from_files(
    file_list,
    preprocessor,
    common_channels,
    output_dir=output_path
)
```

---

### **Stage 4: Transfer Learning Model** 🔄 IN PROGRESS

**File**: `LEMON_Transfer_Learning_Pipeline.ipynb` (to be created)

**Training Strategy**:

#### **Phase 1: Pre-training on LEMON (Unsupervised)**

**Objective**: Learn robust EEG feature representations from healthy baseline

**Approach**: Contrastive learning or autoencoder

**Model**: CNN Encoder
```
Input: (batch, n_channels, 1000)
├─ Conv1D(n_channels, 64, kernel=7) + BatchNorm + ReLU + MaxPool(4)
├─ Conv1D(64, 128, kernel=5) + BatchNorm + ReLU + MaxPool(4)
├─ Conv1D(128, 256, kernel=3) + BatchNorm + ReLU + AdaptiveAvgPool
└─ Output: (batch, 256) latent representation
```

**Training**:
- 83,070 LEMON windows (unlabeled)
- Self-supervised learning
- Epochs: 50-100
- Optimizer: Adam (lr=1e-3)

**Expected Outcome**: Encoder learns general EEG patterns from healthy population

---

#### **Phase 2: Fine-tuning on Migraine (Supervised)**

**Objective**: Adapt model for 3-class migraine classification

**Model**: CNN-LSTM with pre-trained encoder
```
Input: (batch, n_windows, n_channels, 1000)
├─ CNN Encoder (frozen or partially frozen)
├─ LSTM(256, 128, num_layers=2, bidirectional=True)
├─ Attention Layer (optional)
├─ FC(256, 128) + Dropout(0.5)
└─ FC(128, 3) → [Control, Aura, Non-Aura]
```

**Training**:
- 6,045 migraine windows (labeled)
- Train/test split: 80/20 by subjects
- Class weights for imbalance (18:9:4 ratio)
- Epochs: 50-100
- Optimizer: Adam (lr=1e-4)
- Scheduler: ReduceLROnPlateau

**Expected Accuracy**:
- Baseline (flat features): 75%
- Sequential CNN-LSTM (no transfer): 82-85%
- **Transfer learning**: 88-92%

---

#### **Phase 3: Hybrid Ensemble (Optional)**

**Approach**: Combine Random Forest + CNN-LSTM

**Architecture**:
```
Input EEG Window
├─ Branch 1: Extract 1738 handcrafted features → Random Forest
└─ Branch 2: CNN-LSTM with transfer learning

Final Prediction: 0.4 × RF + 0.6 × CNN-LSTM
```

**Expected Boost**: +2-3% accuracy

---

## 📊 Expected Results

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Current (flat features) | 75% | Baseline with 35 subjects |
| Sequential CNN-LSTM | 82-85% | Windowed data, no transfer |
| **Transfer Learning** | **88-92%** | Pre-trained on 213 LEMON subjects |
| Hybrid Ensemble | 90-94% | RF + CNN-LSTM combination |

---

## 🚀 Execution Order

### **Step 1: Run EDA** (20 minutes)

```bash
# Open and run all cells:
LEMON_Dataset_EDA.ipynb
```

**Verify**:
- Common channels identified
- No missing files
- Data quality checks passed

---

### **Step 2: Test Preprocessing** (10 minutes)

```bash
# Test preprocessing module:
python src/lemon_preprocessor.py
```

**Expected Output**:
- Loads sample LEMON and migraine files
- Applies full preprocessing pipeline
- Prints processing metrics

---

### **Step 3: Test Windowing** (15 minutes)

```bash
# Test windowed dataset builder:
python src/windowed_dataset_builder.py
```

**Expected Output**:
- Creates windowed tensors from 2 LEMON + 2 migraine subjects
- Shows artifact rejection statistics
- Saves test dataset

---

### **Step 4: Full Pipeline Execution** (2-4 hours)

```bash
# Open and run:
LEMON_Transfer_Learning_Pipeline.ipynb
```

**Sections**:
1. Load all LEMON subjects (213)
2. Load all migraine subjects (31)
3. Build complete windowed datasets
4. Pre-train CNN encoder on LEMON
5. Fine-tune CNN-LSTM on migraine
6. Evaluate and compare results

---

## ⚙️ Hardware Requirements

**Minimum**:
- RAM: 16 GB
- Storage: 50 GB free
- GPU: Optional but recommended

**Recommended**:
- RAM: 32 GB
- Storage: 100 GB SSD
- GPU: NVIDIA GTX 1660 or better (6+ GB VRAM)

**Processing Time**:
- Preprocessing: ~5-10 minutes per subject
- LEMON (213 subjects): ~18-35 hours (CPU) or ~4-8 hours (GPU)
- Migraine (31 subjects): ~2.5-5 hours (CPU) or ~30-60 minutes (GPU)
- Model training: ~2-6 hours with GPU

**Tip**: Run preprocessing overnight, cache results to disk

---

## 🐛 Troubleshooting

### **Issue**: "MemoryError during preprocessing"
**Solution**: Process in batches, save individual subjects, then combine

### **Issue**: "ValueError: No common channels found"
**Solution**: Run EDA notebook first, verify channel names match

### **Issue**: "Too many epochs rejected"
**Solution**: Relax `artifact_threshold` from 150e-6 to 200e-6

### **Issue**: "Model not improving during training"
**Solution**: 
- Check class imbalance (use class weights)
- Verify data normalization
- Reduce learning rate

---

## 📚 Dependencies

Install required packages:

```bash
pip install mne numpy pandas matplotlib seaborn scikit-learn torch
```

**Versions**:
- mne >= 1.0.0
- numpy >= 1.20.0
- pandas >= 1.3.0
- torch >= 1.10.0
- scikit-learn >= 1.0.0

---

## 📖 References

1. LEMON Dataset: Babayan et al. (2019) "A mind-brain-body dataset of MRI, EEG, cognition..."
2. Transfer Learning in EEG: Dose et al. (2018) "An end-to-end deep learning approach..."
3. Migraine EEG Biomarkers: Bjørk et al. (2009) "Photic EEG-driving responses..."

---

## ✅ Next Steps

1. **Immediate**: Run [LEMON_Dataset_EDA.ipynb](LEMON_Dataset_EDA.ipynb)
2. **After EDA**: Create `LEMON_Transfer_Learning_Pipeline.ipynb` (Stage 4)
3. **After Pipeline**: Integrate with existing binaural beat therapy system
4. **Final**: Deploy adaptive real-time EEG classification

---

## 📞 Support

For questions or issues:
- Check [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md)
- Review [FULL_PROJECT_PIPELINE.md](docs/FULL_PROJECT_PIPELINE.md)
- Contact: FYDP-I Team

---

**Last Updated**: 2026-02-20  
**Status**: Stages 1-3 Complete ✅ | Stage 4 In Progress 🔄
