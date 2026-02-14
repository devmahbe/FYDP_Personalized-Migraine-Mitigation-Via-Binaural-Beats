"""
Data Loader Module
Loads clinical demographics and EEG data from .bdf files
"""
import os
import pandas as pd
import mne
import numpy as np
from pathlib import Path

# Base directory for dataset
BASE_DIR = Path("EEG Dataset and Migrain Patient")

# Patients to exclude based on README
EXCLUDED_PATIENTS = ['M2', 'M6', 'M13', 'M18']


def load_clinical_data(exclude_problematic=True):
    """
    Load clinical demographics from Excel file
    
    Args:
        exclude_problematic: If True, exclude patients with medication or missing data
        
    Returns:
        pandas.DataFrame with clinical information
    """
    filepath = BASE_DIR / "Migraine_Control_Demographics.xlsx"
    df = pd.read_excel(filepath)
    
    if exclude_problematic:
        # Filter out excluded patients
        df = df[~df['P#'].isin(EXCLUDED_PATIENTS)]
    
    # Add label column: 0=Control, 1=Aura, 2=Non-Aura
    def get_label(row):
        patient_id = row['P#']
        if patient_id.startswith('C'):
            return 0  # Control
        elif row['Aura?'] == 'Yes':
            return 1  # Migraine with Aura
        else:
            return 2  # Migraine without Aura
    
    df['label'] = df.apply(get_label, axis=1)
    
    # Clean gender encoding
    df['gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    return df


def load_eeg_file(patient_id, task_type='resting', verbose=False):
    """
    Load EEG data from .bdf file using MNE
    
    Args:
        patient_id: Patient identifier (e.g., 'M1_1', 'C1')
        task_type: Type of recording ('resting', 'SSAEP', 'SSVEP')
        verbose: Print loading information
        
    Returns:
        MNE Raw object containing EEG data
    """
    # Determine the correct folder name
    folder_name = patient_id
    base_name = patient_id.split('_')[0]  # M1 from M1_1, C1 from C1
    
    # Build file path based on task type
    if task_type == 'resting':
        # Try multiple naming conventions for resting state
        possible_names = [
            f"{base_name}resting.bdf",        # M1resting.bdf
            f"{base_name}Resting.bdf",        # M3Resting.bdf
            f"{base_name}_Resting.bdf",       # M10_Resting.bdf, C1_Resting.bdf
        ]
    elif task_type == 'SSAEP':
        possible_names = [f"{base_name}_SSAEP.bdf"]
    elif task_type == 'SSVEP':
        possible_names = [f"{base_name}_SSVEP.bdf"]
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Try each possible filename
    filepath = None
    for filename in possible_names:
        test_path = BASE_DIR / folder_name / filename
        if test_path.exists():
            filepath = test_path
            break
    
    if filepath is None:
        raise FileNotFoundError(f"EEG file not found for {patient_id} - {task_type}. Tried: {possible_names}")
    
    # Load using MNE
    # Set preload=True to load data into memory
    raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=verbose)
    
    return raw


def load_all_tasks(patient_id, verbose=False):
    """
    Load all available EEG tasks for comprehensive multi-task analysis
    
    Args:
        patient_id: Patient identifier (e.g., 'M1_1', 'C1')
        verbose: Print loading information
        
    Returns:
        Dictionary with task data: {'resting': raw, 'SSAEP': raw, 'SSVEP': raw}
        Missing tasks will have None value
    """
    tasks = {}
    task_types = ['resting', 'SSAEP', 'SSVEP']
    
    for task in task_types:
        try:
            tasks[task] = load_eeg_file(patient_id, task, verbose=False)
            if verbose:
                print(f"  ✓ Loaded {task}")
        except FileNotFoundError:
            tasks[task] = None
            if verbose:
                print(f"  ✗ {task} not found")
    
    return tasks


def get_patient_label(patient_id, clinical_df):
    """
    Get classification label for a patient
    
    Args:
        patient_id: Patient identifier (e.g., 'M1_1', 'C1')
        clinical_df: DataFrame from load_clinical_data()
        
    Returns:
        int: 0=Control, 1=Aura, 2=Non-Aura
    """
    # Extract base patient ID (M1 from M1_1)
    base_id = patient_id.split('_')[0]
    
    row = clinical_df[clinical_df['P#'] == base_id]
    
    if len(row) == 0:
        raise ValueError(f"Patient {base_id} not found in clinical data")
    
    return row['label'].values[0]


def get_all_patient_ids(exclude_problematic=True):
    """
    Get list of all patient IDs with available data
    
    Returns:
        list: Patient IDs (e.g., ['M1_1', 'M3_2', 'C1', ...])
    """
    patient_ids = []
    
    # Get all directories in base directory
    for item in BASE_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it's a patient directory (starts with M or C)
            if item.name.startswith('M') or item.name.startswith('C'):
                # Extract base patient ID
                base_id = item.name.split('_')[0]
                
                # Skip excluded patients
                if exclude_problematic and base_id in EXCLUDED_PATIENTS:
                    continue
                
                patient_ids.append(item.name)
    
    return sorted(patient_ids)


if __name__ == "__main__":
    """Test the data loader"""
    print("=" * 60)
    print("Testing Data Loader Module")
    print("=" * 60)
    
    # Test clinical data loading
    print("\n1. Loading clinical data...")
    clinical_df = load_clinical_data()
    print(f"Loaded {len(clinical_df)} patients")
    print(f"\nLabel distribution:")
    print(clinical_df['label'].value_counts().sort_index())
    print(f"\nFirst few rows:")
    print(clinical_df[['P#', 'Gender', 'Age', 'Aura?', 'label']].head(10))
    
    # Test EEG loading
    print("\n2. Testing EEG data loading...")
    try:
        raw = load_eeg_file('M1_1', 'resting', verbose=False)
        print(f"✓ Successfully loaded M1_1 resting EEG")
        print(f"  - Channels: {len(raw.ch_names)}")
        print(f"  - Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  - Duration: {raw.times[-1]:.1f} seconds")
        print(f"  - Data shape: {raw.get_data().shape}")
    except Exception as e:
        print(f"✗ Error loading EEG: {e}")
    
    # Test get all patient IDs
    print("\n3. Getting all patient IDs...")
    patient_ids = get_all_patient_ids()
    print(f"Found {len(patient_ids)} valid patients")
    print(f"Migraine patients: {[p for p in patient_ids if p.startswith('M')]}")
    print(f"Control patients: {[p for p in patient_ids if p.startswith('C')]}")
    
    print("\n" + "=" * 60)
    print("Data Loader Test Complete!")
    print("=" * 60)
