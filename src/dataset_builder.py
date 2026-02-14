"""
Dataset Builder Module
Builds complete training dataset from all patients
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_clinical_data, get_all_patient_ids
from feature_extraction import extract_all_features


def build_dataset(task='resting', output_dir='data', verbose=True):
    """
    Build complete dataset with features from all patients
    
    Args:
        task: EEG task to use ('resting', 'SSAEP', 'SSVEP')
        output_dir: Directory to save processed data
        verbose: Print progress
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,) - 0=Control, 1=Aura, 2=Non-Aura
        metadata: DataFrame with patient info
    """
    if verbose:
        print("=" * 70)
        print(f"Building Dataset from {task.upper()} EEG")
        print("=" * 70)
    
    # Load clinical data
    clinical_df = load_clinical_data(exclude_problematic=True)
    
    # Get all patient IDs
    patient_ids = get_all_patient_ids(exclude_problematic=True)
    
    if verbose:
        print(f"\nTotal patients to process: {len(patient_ids)}")
        print(f"  - Migraine: {len([p for p in patient_ids if p.startswith('M')])}")
        print(f"  - Control: {len([p for p in patient_ids if p.startswith('C')])}")
    
    # Extract features for each patient
    X_list = []
    y_list = []
    metadata_list = []
    
    for idx, patient_id in enumerate(patient_ids):
        if verbose:
            print(f"\n[{idx+1}/{len(patient_ids)}] Processing {patient_id}...")
        
        # Extract EEG features
        features = extract_all_features(patient_id, task=task, verbose=verbose)
        
        if features is None:
            if verbose:
                print(f"  ⚠ Skipping {patient_id} (missing data)")
            continue
        
        # Get label
        base_id = patient_id.split('_')[0]
        patient_row = clinical_df[clinical_df['P#'] == base_id]
        
        if len(patient_row) == 0:
            if verbose:
                print(f"  ⚠ Skipping {patient_id} (no clinical data)")
            continue
        
        label = patient_row['label'].values[0]
        age = patient_row['Age'].values[0]
        gender_encoded = patient_row['gender_encoded'].values[0]
        
        # Add clinical features to EEG features
        clinical_features = np.array([age, gender_encoded])
        combined_features = np.concatenate([features, clinical_features])
        
        X_list.append(combined_features)
        y_list.append(label)
        metadata_list.append({
            'patient_id': patient_id,
            'label': label,
            'age': age,
            'gender': patient_row['Gender'].values[0]
        })
        
        if verbose:
            print(f"  ✓ Features: {len(combined_features)}, Label: {label}")
    
    # Convert to numpy arrays
    X = np.array(X_list)
    y = np.array(y_list)
    metadata = pd.DataFrame(metadata_list)
    
    if verbose:
        print("\n" + "=" * 70)
        print("Dataset Building Complete!")
        print("=" * 70)
        print(f"\nFinal Dataset:")
        print(f"  - Total samples: {X.shape[0]}")
        print(f"  - Features per sample: {X.shape[1]}")
        print(f"  - Label distribution:")
        for label_val in [0, 1, 2]:
            count = np.sum(y == label_val)
            label_name = ['Control', 'Aura', 'Non-Aura'][label_val]
            print(f"      {label_name} ({label_val}): {count}")
    
    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    dataset = {
        'X': X,
        'y': y,
        'metadata': metadata,
        'task': task
    }
    
    save_path = output_path / f'dataset_{task}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    if verbose:
        print(f"\n✓ Dataset saved to: {save_path}")
    
    return X, y, metadata


def load_dataset(task='resting', data_dir='data'):
    """
    Load pre-built dataset
    
    Args:
        task: EEG task type
        data_dir: Directory containing saved data
        
    Returns:
        X, y, metadata
    """
    load_path = Path(data_dir) / f'dataset_{task}.pkl'
    
    if not load_path.exists():
        raise FileNotFoundError(f"Dataset not found: {load_path}\nPlease run build_dataset() first.")
    
    with open(load_path, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset['X'], dataset['y'], dataset['metadata']


if __name__ == "__main__":
    """Build the dataset"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='resting', 
                       help='EEG task type (resting, SSAEP, SSVEP)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory')
    args = parser.parse_args()
    
    # Build dataset
    X, y, metadata = build_dataset(task=args.task, output_dir=args.output_dir, verbose=True)
    
    print("\n" + "=" * 70)
    print("Dataset saved successfully!")
    print("To load: from dataset_builder import load_dataset")
    print(f"         X, y, metadata = load_dataset(task='{args.task}')")
    print("=" * 70)
