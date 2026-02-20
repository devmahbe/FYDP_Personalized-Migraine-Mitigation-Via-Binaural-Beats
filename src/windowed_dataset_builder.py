"""
Windowed Dataset Builder for Transfer Learning

This module creates 4-second windowed tensors from preprocessed EEG data with:
- 50% overlap (2-second stride)
- Automatic artifact rejection
- Per-channel per-subject z-score normalization
- Subject-level train/test splits (no data leakage)

Outputs tensor shape: (n_samples, n_channels, n_timepoints)

Author: FYDP-I Migraine Detection Team
Date: 2026-02-20
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pickle
import warnings
warnings.filterwarnings('ignore')


class WindowedDatasetBuilder:
    """
    Creates windowed EEG tensors from preprocessed continuous recordings.
    Handles both labeled (migraine) and unlabeled (LEMON) datasets.
    """
    
    def __init__(
        self,
        window_duration: float = 4.0,
        overlap: float = 0.5,
        artifact_threshold: float = 150e-6,  # 150 µV
        min_epochs_per_subject: int = 10,
        normalize_per_subject: bool = True,
        verbose: bool = True
    ):
        """
        Initialize windowed dataset builder.
        
        Args:
            window_duration: Window size in seconds
            overlap: Overlap fraction (0.5 = 50%)
            artifact_threshold: Peak-to-peak amplitude threshold for rejection (Volts)
            min_epochs_per_subject: Minimum clean epochs required per subject
            normalize_per_subject: Apply z-score normalization per channel per subject
            verbose: Print processing information
        """
        self.window_duration = window_duration
        self.overlap = overlap
        self.artifact_threshold = artifact_threshold
        self.min_epochs_per_subject = min_epochs_per_subject
        self.normalize_per_subject = normalize_per_subject
        self.verbose = verbose
        
    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"  {message}")
    
    def create_epochs_from_raw(
        self,
        raw: mne.io.Raw,
        reject_artifacts: bool = True
    ) -> Tuple[mne.Epochs, Dict]:
        """
        Segment continuous raw data into overlapping epochs.
        
        Args:
            raw: Preprocessed MNE Raw object
            reject_artifacts: Apply amplitude-based artifact rejection
            
        Returns:
            (epochs_object, metadata_dict)
        """
        sfreq = raw.info['sfreq']
        duration = raw.times[-1]
        
        # Calculate stride
        stride = self.window_duration * (1 - self.overlap)
        
        # Calculate number of windows
        n_windows = int((duration - self.window_duration) / stride) + 1
        
        self.log(f"Creating {n_windows} epochs (window={self.window_duration}s, stride={stride}s)")
        
        # Create fixed-length epochs
        events = mne.make_fixed_length_events(
            raw,
            duration=self.window_duration,
            overlap=self.window_duration - stride
        )
        
        # Define rejection criteria
        reject_dict = None
        if reject_artifacts:
            reject_dict = {'eeg': self.artifact_threshold}  # Peak-to-peak amplitude
        
        # Create epochs
        epochs = mne.Epochs(
            raw,
            events=events,
            tmin=0,
            tmax=self.window_duration - 1/sfreq,  # Exclude last sample
            baseline=None,  # No baseline correction (already preprocessed)
            reject=reject_dict,
            preload=True,
            verbose=False
        )
        
        n_rejected = n_windows - len(epochs)
        accept_rate = len(epochs) / n_windows * 100
        
        metadata = {
            'n_windows_total': n_windows,
            'n_windows_accepted': len(epochs),
            'n_windows_rejected': n_rejected,
            'acceptance_rate': accept_rate,
            'window_duration': self.window_duration,
            'stride': stride
        }
        
        self.log(f"✓ Created {len(epochs)} clean epochs ({accept_rate:.1f}% acceptance)")
        if n_rejected > 0:
            self.log(f"  Rejected {n_rejected} epochs due to artifacts")
        
        return epochs, metadata
    
    def normalize_epochs(
        self,
        data: np.ndarray,
        method: str = 'zscore'
    ) -> np.ndarray:
        """
        Normalize epoch data per channel.
        
        Args:
            data: Shape (n_epochs, n_channels, n_samples)
            method: 'zscore' or 'minmax'
            
        Returns:
            Normalized data with same shape
        """
        if method == 'zscore':
            # Z-score per channel across all epochs
            mean = np.mean(data, axis=(0, 2), keepdims=True)  # (1, n_channels, 1)
            std = np.std(data, axis=(0, 2), keepdims=True) + 1e-8
            data_norm = (data - mean) / std
        elif method == 'minmax':
            # Min-max per channel
            min_val = np.min(data, axis=(0, 2), keepdims=True)
            max_val = np.max(data, axis=(0, 2), keepdims=True)
            data_norm = (data - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return data_norm
    
    def process_single_subject(
        self,
        raw: mne.io.Raw,
        subject_id: str,
        label: Optional[int] = None,
        dataset_type: str = 'lemon'
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Process single subject: create epochs and extract windowed data.
        
        Args:
            raw: Preprocessed MNE Raw object
            subject_id: Subject identifier
            label: Class label (None for unlabeled LEMON data)
            dataset_type: 'lemon' or 'migraine'
            
        Returns:
            (windowed_data, metadata_dataframe)
            windowed_data shape: (n_windows, n_channels, n_samples)
        """
        self.log(f"Processing subject: {subject_id}")
        
        # Create epochs
        epochs, epoch_meta = self.create_epochs_from_raw(raw, reject_artifacts=True)
        
        # Check minimum epochs requirement
        if len(epochs) < self.min_epochs_per_subject:
            raise ValueError(
                f"Subject {subject_id} has only {len(epochs)} clean epochs "
                f"(minimum required: {self.min_epochs_per_subject})"
            )
        
        # Get epoch data: (n_epochs, n_channels, n_samples)
        data = epochs.get_data()
        
        # Normalize per subject
        if self.normalize_per_subject:
            data = self.normalize_epochs(data, method='zscore')
            self.log(f"✓ Z-score normalized per channel")
        
        # Create metadata for each window
        metadata_records = []
        for epoch_idx in range(len(data)):
            metadata_records.append({
                'subject_id': subject_id,
                'dataset_type': dataset_type,
                'epoch_idx': epoch_idx,
                'label': label,
                'n_channels': data.shape[1],
                'n_samples': data.shape[2],
                'sampling_rate': epochs.info['sfreq']
            })
        
        metadata_df = pd.DataFrame(metadata_records)
        
        return data, metadata_df
    
    def build_dataset_from_files(
        self,
        file_list: List[Tuple[Path, str, str, Optional[int]]],
        preprocessor,
        target_channels: List[str],
        output_dir: Optional[Path] = None,
        save_individual: bool = False
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Build complete windowed dataset from list of files.
        
        Args:
            file_list: List of (file_path, subject_id, dataset_type, label)
            preprocessor: UnifiedEEGPreprocessor instance
            target_channels: Common channel list for alignment
            output_dir: Directory to save processed data (None = don't save)
            save_individual: Save individual subject data
            
        Returns:
            (all_windowed_data, all_metadata)
            all_windowed_data shape: (total_windows, n_channels, n_samples)
        """
        if self.verbose:
            print("\n" + "="*60)
            print(f"BUILDING WINDOWED DATASET")
            print("="*60)
            print(f"Total files to process: {len(file_list)}")
            print(f"Window size: {self.window_duration}s")
            print(f"Overlap: {self.overlap * 100}%")
            print(f"Artifact threshold: {self.artifact_threshold * 1e6:.1f} µV")
            print("="*60 + "\n")
        
        all_data = []
        all_metadata = []
        failed_subjects = []
        
        for idx, (file_path, subject_id, dataset_type, label) in enumerate(file_list, 1):
            try:
                if self.verbose:
                    print(f"\n[{idx}/{len(file_list)}] Processing: {subject_id}")
                    print("-" * 60)
                
                # Preprocess file
                raw_clean, prep_meta = preprocessor.preprocess_file(
                    file_path,
                    dataset_type=dataset_type,
                    target_channels=target_channels,
                    eog_channels=None  # EOG already removed in preprocessing
                )
                
                # Create windowed data
                windowed_data, window_meta = self.process_single_subject(
                    raw_clean,
                    subject_id,
                    label,
                    dataset_type
                )
                
                # Store
                all_data.append(windowed_data)
                all_metadata.append(window_meta)
                
                # Save individual if requested
                if save_individual and output_dir:
                    subject_output = output_dir / 'individual' / subject_id
                    subject_output.mkdir(parents=True, exist_ok=True)
                    
                    np.save(subject_output / 'windowed_data.npy', windowed_data)
                    window_meta.to_csv(subject_output / 'metadata.csv', index=False)
                
                if self.verbose:
                    print(f"✓ Successfully processed {subject_id}: {len(windowed_data)} windows")
                
            except Exception as e:
                print(f"❌ Failed to process {subject_id}: {e}")
                failed_subjects.append((subject_id, str(e)))
                continue
        
        # Concatenate all data
        if not all_data:
            raise RuntimeError("No subjects were successfully processed!")
        
        all_windowed_data = np.concatenate(all_data, axis=0)
        all_metadata_df = pd.concat(all_metadata, ignore_index=True)
        
        # Summary
        if self.verbose:
            print("\n" + "="*60)
            print("DATASET BUILD COMPLETE")
            print("="*60)
            print(f"✓ Successfully processed: {len(all_data)} subjects")
            print(f"❌ Failed: {len(failed_subjects)} subjects")
            print(f"\nDataset shape: {all_windowed_data.shape}")
            print(f"  - Total windows: {all_windowed_data.shape[0]:,}")
            print(f"  - Channels: {all_windowed_data.shape[1]}")
            print(f"  - Samples per window: {all_windowed_data.shape[2]}")
            
            # Label distribution
            if all_metadata_df['label'].notna().any():
                label_counts = all_metadata_df['label'].value_counts().sort_index()
                print(f"\nLabel distribution:")
                for label, count in label_counts.items():
                    print(f"  Label {int(label)}: {count:,} windows")
            
            if failed_subjects:
                print(f"\n⚠️ Failed subjects:")
                for subj, error in failed_subjects[:5]:  # Show first 5
                    print(f"  - {subj}: {error}")
                if len(failed_subjects) > 5:
                    print(f"  ... and {len(failed_subjects) - 5} more")
            
            print("="*60 + "\n")
        
        return all_windowed_data, all_metadata_df
    
    def save_dataset(
        self,
        data: np.ndarray,
        metadata: pd.DataFrame,
        output_dir: Path,
        dataset_name: str = 'windowed_dataset'
    ):
        """
        Save windowed dataset to disk.
        
        Args:
            data: Windowed data array
            metadata: Metadata DataFrame
            output_dir: Output directory
            dataset_name: Name prefix for saved files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_path = output_dir / f'{dataset_name}.npy'
        np.save(data_path, data)
        self.log(f"✓ Saved data: {data_path}")
        
        # Save metadata
        meta_path = output_dir / f'{dataset_name}_metadata.csv'
        metadata.to_csv(meta_path, index=False)
        self.log(f"✓ Saved metadata: {meta_path}")
        
        # Save dataset info
        info = {
            'data_shape': data.shape,
            'window_duration': self.window_duration,
            'overlap': self.overlap,
            'artifact_threshold': self.artifact_threshold,
            'n_subjects': metadata['subject_id'].nunique(),
            'n_total_windows': len(data),
            'dataset_types': metadata['dataset_type'].unique().tolist(),
        }
        
        if metadata['label'].notna().any():
            info['label_distribution'] = metadata['label'].value_counts().to_dict()
        
        info_path = output_dir / f'{dataset_name}_info.pkl'
        with open(info_path, 'wb') as f:
            pickle.dump(info, f)
        self.log(f"✓ Saved dataset info: {info_path}")
        
        print(f"\n✓ Dataset saved to: {output_dir}")
    
    @staticmethod
    def load_dataset(
        output_dir: Path,
        dataset_name: str = 'windowed_dataset'
    ) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
        """
        Load saved windowed dataset.
        
        Args:
            output_dir: Directory containing saved dataset
            dataset_name: Name prefix of saved files
            
        Returns:
            (data, metadata, info_dict)
        """
        data = np.load(output_dir / f'{dataset_name}.npy')
        metadata = pd.read_csv(output_dir / f'{dataset_name}_metadata.csv')
        
        with open(output_dir / f'{dataset_name}_info.pkl', 'rb') as f:
            info = pickle.load(f)
        
        print(f"✓ Loaded dataset from: {output_dir}")
        print(f"  Shape: {data.shape}")
        print(f"  Subjects: {info['n_subjects']}")
        print(f"  Total windows: {info['n_total_windows']:,}")
        
        return data, metadata, info


def create_train_test_split_by_subject(
    data: np.ndarray,
    metadata: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by subjects (not epochs) to avoid data leakage.
    
    Args:
        data: Windowed data (n_windows, n_channels, n_samples)
        metadata: Metadata DataFrame
        test_size: Fraction of subjects for test set
        random_state: Random seed
        
    Returns:
        (X_train, X_test, y_train, y_test, meta_train, meta_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Get unique subjects
    subjects = metadata['subject_id'].unique()
    
    # Get labels per subject (for stratification if labeled)
    subject_labels = metadata.groupby('subject_id')['label'].first()
    
    # Split subjects
    if subject_labels.notna().all():
        # Labeled dataset: stratified split
        train_subjects, test_subjects = train_test_split(
            subjects,
            test_size=test_size,
            random_state=random_state,
            stratify=subject_labels
        )
    else:
        # Unlabeled dataset: random split
        train_subjects, test_subjects = train_test_split(
            subjects,
            test_size=test_size,
            random_state=random_state
        )
    
    # Split data by subject membership
    train_mask = metadata['subject_id'].isin(train_subjects)
    test_mask = metadata['subject_id'].isin(test_subjects)
    
    X_train = data[train_mask]
    X_test = data[test_mask]
    
    y_train = metadata.loc[train_mask, 'label'].values
    y_test = metadata.loc[test_mask, 'label'].values
    
    meta_train = metadata[train_mask].reset_index(drop=True)
    meta_test = metadata[test_mask].reset_index(drop=True)
    
    print(f"\n📊 Train/Test Split (by subject):")
    print(f"  Train: {len(train_subjects)} subjects, {len(X_train):,} windows")
    print(f"  Test: {len(test_subjects)} subjects, {len(X_test):,} windows")
    
    if np.isnan(y_train).sum() == 0:  # Labeled data
        print(f"\n  Train label distribution: {np.bincount(y_train.astype(int))}")
        print(f"  Test label distribution: {np.bincount(y_test.astype(int))}")
    
    return X_train, X_test, y_train, y_test, meta_train, meta_test


if __name__ == "__main__":
    """
    Example usage
    """
    from pathlib import Path
    from lemon_preprocessor import UnifiedEEGPreprocessor, get_common_channels_between_datasets
    
    # Paths
    lemon_path = Path(r'g:\Study\FYDP-I_Personalized-Migraine-Mitigation-Via-Binaural-Beats\EEG_MPILMBB_LEMON\EEG_Raw_BIDS_ID')
    migraine_path = Path(r'g:\Study\FYDP-I_Personalized-Migraine-Mitigation-Via-Binaural-Beats\Dataset')
    output_path = Path(r'g:\Study\FYDP-I_Personalized-Migraine-Mitigation-Via-Binaural-Beats\preprocessed_data')
    
    # Get common channels
    lemon_sample = lemon_path / 'sub-010002' / 'RSEEG' / 'sub-010002.vhdr'
    migraine_sample = migraine_path / 'C1' / 'C1' / 'C1_Resting.bdf'
    common_channels = get_common_channels_between_datasets(lemon_sample, migraine_sample, verbose=False)
    
    print(f"Common channels: {len(common_channels)}")
    
    # Initialize
    preprocessor = UnifiedEEGPreprocessor(verbose=False)
    builder = WindowedDatasetBuilder(
        window_duration=4.0,
        overlap=0.5,
        artifact_threshold=150e-6,
        verbose=True
    )
    
    # Create small test file list (2 LEMON + 2 Migraine)
    file_list = [
        (lemon_path / 'sub-010002' / 'RSEEG' / 'sub-010002.vhdr', 'sub-010002', 'lemon', None),
        (lemon_path / 'sub-010003' / 'RSEEG' / 'sub-010003.vhdr', 'sub-010003', 'lemon', None),
        (migraine_path / 'C1' / 'C1' / 'C1_Resting.bdf', 'C1', 'migraine', 0),  # Control
        (migraine_path / 'M1_1' / 'M1_1' / 'M1_1_Resting.bdf', 'M1_1', 'migraine', 1),  # Aura
    ]
    
    # Build dataset
    data, metadata = builder.build_dataset_from_files(
        file_list,
        preprocessor,
        common_channels,
        output_dir=None,
        save_individual=False
    )
    
    print(f"\n✓ Test complete!")
    print(f"  Final shape: {data.shape}")
    print(f"  Metadata shape: {metadata.shape}")
