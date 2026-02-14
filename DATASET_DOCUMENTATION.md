# Dataset Documentation: Ultra High-Density EEG Recording of Interictal Migraine and Controls

## Overview

This dataset contains **128-channel high-density EEG recordings** from:
- **21 healthy control subjects** (C1-C21)
- **18 migraine patients** (M1-M18) recorded during interictal period

**Data Collection Period**: January 2018 - December 2018  
**Recording System**: 128-channel HD-EEG with customized electrode locations  
**Source**: Chamanzar, Haigh, Grover, and Behrmann (2020), "Abnormalities in cortical pattern of coherence in migraine detected using ultra high-density EEG"

---

## Dataset Structure

### Subject Naming Convention
- **C** prefix: Healthy control subjects (C1-C21)
- **M** prefix: Migraine patients (M1_1 - M18_1)
- Suffix `_1` or `_2`: Indicates migraine subtype
  - `_1`: Migraine **without aura**
  - `_2`: Migraine **with aura**

### Subject Distribution

#### Healthy Controls (21 subjects)
C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21

**Note**: According to README, matched controls used in analysis exclude C2, C6, and C12 (available but not used in original study).

#### Migraine Patients (18 subjects)

**Without Aura (_1):**
M1_1, M4_1, M5_1, M6_1, M7_1, M10_1, M11_1, M14_1, M16_1, M17_1, M18_1 (11 subjects)

**With Aura (_2):**
M3_2, M8_2, M9_2, M12_2, M13_2, M15_2 (6 subjects)

**Excluded from Original Analysis:**
- M2, M6, M18 (on medication)
- M13 (missing auditory recording - SSAEP file)

---

## File Types in Dataset

### 1. **BDF Files** (`.bdf`)
**Format**: BioSemi Data Format - raw EEG recording files  
**Content**: 128-channel continuous EEG data  
**Software Required**: EEGLAB toolbox (MATLAB) or MNE-Python

#### Three experimental conditions per subject:

**a) Resting State**
- **File naming**: `{SubjectID}_Resting.bdf` or `{SubjectID}resting.bdf`
- **Description**: Baseline EEG recorded during eyes-closed or eyes-open rest
- **Examples**: `C1_Resting.bdf`, `M1resting.bdf`, `M3Resting.bdf`
- **Note**: Naming inconsistencies exist (underscore vs no underscore, capitalization)

**b) SSAEP (Steady-State Auditory Evoked Potential)**
- **File naming**: `{SubjectID}_SSAEP.bdf`
- **Description**: EEG during auditory stimulation with modulated tones
- **Frequencies**: 4 Hz and 6 Hz temporal frequencies
- **Examples**: `C1_SSAEP.bdf`, `M1_SSAEP.bdf`
- **Special case**: C14 has two SSAEP recordings (`C14_SSAEP.bdf`, `C14_SSAEP_2.bdf`)
- **Missing**: M13_SSAEP.bdf (subject excluded due to this)

**c) SSVEP (Steady-State Visual Evoked Potential)**
- **File naming**: `{SubjectID}_SSVEP.bdf`
- **Description**: EEG during visual stimulation with vertical grating patterns
- **Frequencies**: 4 Hz and 6 Hz temporal frequencies
- **Examples**: `C1_SSVEP.bdf`, `M1_SSVEP.bdf`

### 2. **TXT Files** (`.txt`)
**Format**: Plain text (tab or comma separated)  
**Content**: Behavioral response data from feedback tasks

**a) Auditory Feedback**
- **File naming**: `{SubjectID}aud_migraine.txt`
- **Description**: Response times (in seconds) during auditory stimulation feedback task
- **Format**: 3 rows of 200 values each:
  - Row 1: Stimulus frequency identifier (1 or 2)
  - Row 2: Response correctness (0=correct, 1=error, 2=timeout)
  - Row 3: Response time in seconds (0 if no response)
- **Examples**: `C1aud_migraine.txt`, `M1aud_migraine.txt`
- **Variants**: Some subjects have `_2aud_migraine.txt` (repeated sessions)

**b) Visual Feedback**
- **File naming**: `{SubjectID}vis_migraine.txt`
- **Description**: Response times (in seconds) during visual stimulation feedback task
- **Format**: Same 3-row structure as auditory feedback
- **Examples**: `C1vis_migraine.txt`, `M1vis_migraine.txt`
- **Variants**: Some subjects have `_2vis_migraine.txt` (repeated sessions)

### 3. **XLSX Files** (`.xlsx`)
**Format**: Microsoft Excel spreadsheet  
**Content**: Alternative format for behavioral data (minority of subjects)
- **Examples**: `C8aud_migraine.xlsx`, `M9vis_migraine.xlsx`, `M14aud_migraine.xlsx`
- **Subjects with XLSX**: C8, M9, M13, M14
- **Note**: Likely contains same data as .txt files but in spreadsheet format

### 4. **MATLAB Files** (`.m`, `.mat`)
**Location**: `EEG_Stimuli/EEG_Stimuli/` directory  
**Purpose**: Stimulus presentation code and parameters

**Files:**
- `aud_SSAEP.m`: MATLAB code for auditory (SSAEP) stimulation
- `vis_SSVEP.m`: MATLAB code for visual (SSVEP) stimulation
- `Resting_migraine.m`: MATLAB code for resting state protocol
- `maskandtest_m1.mat`: SSVEP pattern parameters (pattern set 1)
- `maskandtest_m2.mat`: SSVEP pattern parameters (pattern set 2)

### 5. **System Files**
- `.DS_Store`: macOS system files (can be ignored)
- `__MACOSX/` directories: macOS compression artifacts (can be ignored)

---

## Data Completeness Check

### Complete Datasets (Standard 5 files)
Most subjects have:
1. One Resting BDF
2. One SSAEP BDF
3. One SSVEP BDF
4. One auditory feedback TXT
5. One visual feedback TXT

### Anomalies and Special Cases

| Subject | Issue | Files Affected |
|---------|-------|---------------|
| M13_2 | **Missing SSAEP recording** | No `M13_SSAEP.bdf` |
| C14 | Duplicate SSAEP recording | `C14_SSAEP.bdf`, `C14_SSAEP_2.bdf` |
| C3 | Duplicate auditory feedback | `C3aud_migraine.txt`, `C3_2aud_migraine.txt` |
| C7 | Duplicate visual feedback | `C7vis_migraine.txt`, `C7_2vis_migraine.txt` |
| M6_1 | Duplicate auditory feedback | `M6aud_migraine.txt`, `M6_2aud_migraine.txt` |
| M15_2 | Filename inconsistency | `P15_Resting.bdf` (should be M15) |
| M1_1 | Naming inconsistency | `M1resting.bdf` (no underscore) |
| M3_2 | Naming inconsistency | `M3Resting.bdf` (capital R, no underscore) |

---

## Clinical Groups for Analysis

### Group 1: Healthy Controls (18 matched)
Exclude C2, C6, C12 per original study protocol.  
**Final N = 18**

### Group 2: Migraine Without Aura (MwoA)
M1_1, M4_1, M5_1, M7_1, M10_1, M11_1, M14_1, M16_1, M17_1  
(Exclude M2_1, M6_1, M18_1 due to medication)  
**Final N = 9**

### Group 3: Migraine With Aura (MwA)
M3_2, M8_2, M9_2, M12_2, M15_2  
(Exclude M13_2 due to missing SSAEP)  
**Final N = 5**

---

## Technical Specifications

### EEG Recording
- **Channels**: 128 customized electrode locations
- **System**: BioSemi high-density EEG
- **Format**: BDF (24-bit resolution)
- **Sampling Rate**: To be determined from BDF headers (typically 512-2048 Hz)

### Experimental Paradigm
- **Visual Stimuli**: Vertical grating patterns at 4 Hz and 6 Hz
- **Auditory Stimuli**: Modulated tones at 4 Hz and 6 Hz
- **Resting State**: Eyes closed/open (protocol in stimuli folder)
- **Task**: Feedback detection task for both modalities

---

## Recommended Analysis Pipeline

### 1. Data Loading
- Use MNE-Python (`mne.io.read_raw_bdf()`) or EEGLAB (MATLAB)
- Load all three conditions per subject

### 2. Quality Control
- Check for missing files (M13 SSAEP)
- Resolve naming inconsistencies
- Validate channel count (should be 128 + auxiliary)

### 3. Preprocessing
- Bandpass filtering (0.5-100 Hz recommended)
- Notch filtering (50/60 Hz power line noise)
- Re-referencing (average reference or mastoid)
- Artifact removal (ICA for eye blinks, muscle artifacts)

### 4. Group-Level Analysis
- Compare Controls vs MwoA vs MwA
- Analyze resting-state power spectra
- Compute evoked responses (SSAEP, SSVEP)
- Assess coherence and connectivity patterns

---

## References

**Primary Publication:**  
Chamanzar, A., Haigh, S. M., Grover, P., & Behrmann, M. (2020). "Abnormalities in cortical pattern of coherence in migraine detected using ultra high-density EEG". *Human Brain Mapping*.

**Contact:**
- Alireza Chamanzar (achamanz@andrew.cmu.edu) - Carnegie Mellon University
- Sarah M. Haigh (shaigh@unr.edu) - University of Nevada, Reno

---

## Summary Statistics

| Category | Count |
|----------|-------|
| **Total Subjects** | 39 |
| Healthy Controls | 21 (18 matched) |
| Migraine Patients | 18 |
| - Without Aura | 11 (9 analyzed) |
| - With Aura | 6 (5 analyzed) |
| **Total BDF Files** | 115 (3 conditions × 38 subjects + 1 duplicate) |
| **Total Behavioral Files** | ~114 (txt + xlsx) |

---

**Dataset Status**: ✅ Complete with documented anomalies  
**Last Updated**: January 2026
