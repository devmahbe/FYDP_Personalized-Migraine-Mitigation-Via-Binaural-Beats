# Migraine Detection & Personalized Binaural Beat Therapy System

An intelligent system that classifies migraine types (Control, Aura, Non-Aura) from clinical and EEG data, then generates personalized binaural beat therapy based on individual brain patterns.

## 🎯 Features

- **Multi-Class Classification**: 84.6% cross-validation accuracy using Random Forest
- **High-Density EEG Processing**: 128-channel biosignal analysis with 1,738 features
- **Personalized Therapy**: Binaural beats tailored to:
  - Migraine type (Aura vs Non-Aura)
  - Individual EEG abnormalities
  - Patient demographics (age, gender)
- **Complete Pipeline**: From raw EEG data to therapeutic audio in seconds
- **Clinical Documentation**: Detailed treatment reports for transparency

## 📊 Dataset

- **31 patients** total (18 control, 9 aura, 4 non-aura)
- **128-channel EEG** at 512 Hz sampling rate
- **~13 minutes** of resting-state recordings per patient
- **Source**: High-density EEG study of interictal migraine patients

## 🚀 Quick Start

### Installation

```bash
cd /Users/mahmudulmashrafe/Programming/FYDP/3
pip install -r requirements.txt
```

### Process a Single Patient

```bash
python3 src/main_pipeline.py --patient M1_1 --duration 600
```

**Outputs:**
- `output/M1_1_binaural_beat.wav` - 10-minute therapeutic audio
- `output/M1_1_treatment_report.txt` - Detailed treatment explanation

### Batch Processing

```bash
python3 src/main_pipeline.py --batch --duration 300
```

Processes multiple patients (5-minute audio each).

### Interactive Demo

```bash
jupyter notebook migraine_binaural_treatment.ipynb
```

Step-by-step demonstration with visualizations.

## 📁 Project Structure

```
.
├── src/
│   ├── data_loader.py              # Load clinical & EEG data
│   ├── feature_extraction.py       # Extract 1,738 features
│   ├── dataset_builder.py          # Compile dataset
│   ├── classifier.py               # Train/evaluate model
│   ├── binaural_beat_generator.py  # Generate therapy audio
│   └── main_pipeline.py            # End-to-end workflow
├── data/
│   └── dataset_resting.pkl         # Processed dataset
├── models/
│   └── migraine_classifier.pkl     # Trained model
├── output/                         # Generated audio & reports
├── requirements.txt                # Dependencies
├── migraine_binaural_treatment.ipynb  # Demo notebook
└── README.md                       # This file
```

## 🧠 How It Works

### 1. Data Loading
- Reads clinical demographics from Excel
- Loads 128-channel EEG from .bdf files using MNE

### 2. Feature Extraction (1,738 features)
- **Power Spectral Density**: 5 frequency bands × 128 channels
- **Statistical**: Mean, variance, skewness, kurtosis per channel
- **Connectivity**: Coherence between electrode pairs
- **Band Ratios**: Theta/Alpha, Delta/Alpha, etc.

### 3. Classification
- **Preprocessing**: NaN imputation, StandardScaler, PCA (30 components)
- **Model**: Random Forest (200 trees, balanced class weights)
- **Augmentation**: SMOTE oversampling for minority classes

### 4. Binaural Beat Generation

**Personalization Algorithm:**

| Input | Effect on Beat Frequency |
|-------|--------------------------|
| **Migraine Type** | Aura → 10 Hz (Alpha), Non-Aura → 7 Hz (Theta) |
| **High Delta/Theta** | +1-2 Hz (boost toward alpha) |
| **High Beta** | -2 Hz (calm toward theta) |
| **Age < 25** | +0.5 Hz |
| **Age > 40** | -0.5 Hz |
| **Female** | +0.3 Hz |

**Final Output:** Stereo WAV file with carrier frequency in one ear, carrier+beat in the other.

## 📈 Performance

- **Cross-Validation**: 84.6% ± 6.3% accuracy
- **Test Set**: 62.5% accuracy (small test set, n=8)
- **Processing Time**: ~15 seconds per patient
- **Audio Quality**: 44.1 kHz stereo with smooth fade in/out

## 🔬 Sample Results

### Patient M3_2 (Migraine without Aura)
- **Predicted**: Non-Aura (62.3% confidence)
- **EEG**: 99.83% alpha power (very high)
- **Beat Frequency**: 7.8 Hz (Theta band for deep relaxation)
- **Files**: `M3_2_binaural_beat.wav`, `M3_2_treatment_report.txt`

### Batch Processing (5 Control Patients)
| Patient | Age | Beat Freq | EEG Pattern |
|---------|-----|-----------|-------------|
| C1 | 20 | 10.8 Hz | Normal |
| C10 | 43 | 9.5 Hz | Normal |
| C11 | 24 | 12.0 Hz | High Delta |
| C13 | 20 | 11.5 Hz | Normal |
| C14 | 20 | 8.8 Hz | High Beta |

## ⚠️ Important Notes

### Clinical Disclaimer
This is an **experimental system** for research purposes. Binaural beat therapy:
- Should **complement, not replace** conventional medical treatment
- Has **not been clinically validated** in this implementation
- Frequency selection is based on neuroscience literature, not empirical trials
- **Consult a healthcare professional** before use

### Usage Safety
- ✅ Use stereo headphones (required for binaural effect)
- ✅ Listen at comfortable volume
- ✅ Use in quiet, relaxed environment
- ❌ Do NOT use while driving or operating machinery
- ❌ Stop if you experience discomfort

### Dataset Limitations
- Small sample size (31 patients)
- Class imbalance (4 non-aura vs 18 control)
- Test performance may vary due to limited data

## 🔮 Future Enhancements

**Technical:**
- [ ] Incorporate SSAEP/SSVEP task data for richer features
- [ ] Deep learning models (CNN/RNN) for raw EEG
- [ ] Real-time processing for wearable devices

**Clinical:**
- [ ] Larger patient cohort (100+)
- [ ] Longitudinal efficacy studies
- [ ] Collaboration with neurologists for validation

**Deployment:**
- [ ] Mobile app for accessible therapy
- [ ] Adaptive therapy with continuous EEG monitoring
- [ ] Integration with migraine tracking apps

## 📚 References

- **Dataset**: Chamanzar et al. (2020) - "Abnormalities in cortical pattern of coherence in migraine detected using ultra high-density EEG"
- **Binaural Beats**: Based on brainwave entrainment literature (theta/alpha stimulation for migraine relief)

## 📧 Contact

For questions or collaboration: [Your Contact Information]

---

**Developed for FYDP - January 2026**
