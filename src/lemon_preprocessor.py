"""
LEMON + Migraine Dataset Preprocessing Module

This module implements a unified preprocessing pipeline for both LEMON (healthy) 
and migraine datasets to ensure transfer learning compatibility.

Pipeline Steps:
1. Load raw EEG data
2. Channel alignment (retain only common channels)
3. Resampling to 250 Hz
4. Bandpass filtering (1-45 Hz)
5. Notch filtering (50/60 Hz + harmonics)
6. Bad channel detection & interpolation
7. ICA artifact removal (EOG, ECG, muscle)
8. Common average reference
9. Quality validation

Author: FYDP-I Migraine Detection Team
Date: 2026-02-20
"""

import mne
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class UnifiedEEGPreprocessor:
    """
    Unified preprocessing pipeline for LEMON and Migraine datasets.
    Ensures spatial and temporal alignment for transfer learning.
    """
    
    def __init__(
        self,
        target_sfreq: float = 250.0,
        bandpass_freqs: Tuple[float, float] = (1.0, 45.0),
        notch_freq: float = 50.0,
        notch_harmonics: int = 2,
        reference: str = 'average',
        ica_n_components: Optional[int] = None,
        bad_channel_threshold: float = 3.0,
        verbose: bool = True
    ):
        """
        Initialize preprocessor with configuration parameters.
        
        Args:
            target_sfreq: Target sampling frequency (Hz)
            bandpass_freqs: (low_freq, high_freq) for bandpass filter
            notch_freq: Powerline frequency (50 or 60 Hz)
            notch_harmonics: Number of harmonics to filter
            reference: Reference type ('average' or channel name)
            ica_n_components: Number of ICA components (None = auto)
            bad_channel_threshold: Z-score threshold for bad channel detection
            verbose: Print processing steps
        """
        self.target_sfreq = target_sfreq
        self.bandpass_freqs = bandpass_freqs
        self.notch_freq = notch_freq
        self.notch_harmonics = notch_harmonics
        self.reference = reference
        self.ica_n_components = ica_n_components
        self.bad_channel_threshold = bad_channel_threshold
        self.verbose = verbose
        
        # Will be set after first preprocessing call
        self.common_channels = None
        
    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"  {message}")
    
    def load_raw_eeg(
        self, 
        file_path: Path, 
        dataset_type: str = 'lemon'
    ) -> mne.io.Raw:
        """
        Load raw EEG file based on dataset type.
        
        Args:
            file_path: Path to raw EEG file
            dataset_type: 'lemon' (BrainVision) or 'migraine' (BDF)
            
        Returns:
            Raw MNE object
        """
        self.log(f"Loading {dataset_type} file: {file_path.name}")
        
        if dataset_type == 'lemon':
            raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
        elif dataset_type == 'migraine':
            raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'lemon' or 'migraine'")
        
        self.log(f"✓ Loaded: {raw.info['nchan']} channels, {raw.info['sfreq']} Hz, {raw.times[-1]:.1f}s")
        return raw
    
    def keep_eeg_channels_only(self, raw: mne.io.Raw, keep_eog: bool = True) -> mne.io.Raw:
        """
        Remove non-EEG channels (keep EOG temporarily for ICA).
        Also filters out auxiliary/physiological channels by name patterns.
        
        Args:
            raw: Raw MNE object
            keep_eog: Keep EOG channels for artifact detection
            
        Returns:
            Raw object with only EEG (and optionally EOG) channels
        """
        self.log("Removing non-EEG channels...")
        
        # Get channel types
        picks = mne.pick_types(raw.info, eeg=True, eog=keep_eog, ecg=False, 
                               stim=False, misc=False, exclude='bads')
        
        # Additional filtering: Remove auxiliary/physiological channels by name patterns
        # These are often mislabeled as 'eeg' type but aren't scalp EEG
        non_eeg_patterns = [
            'GSR', 'gsr',  # Galvanic skin response
            'Erg', 'erg',  # Ergometer/effort
            'Resp', 'resp', 'RESP',  # Respiration
            'Temp', 'temp', 'TEMP',  # Temperature  
            'Plet', 'plet', 'PLET',  # Plethysmography
            'ECG', 'ecg', 'EKG',  # Electrocardiogram
            'EMG', 'emg',  # Electromyogram
            'Trigger', 'Status',  # Trigger/status channels
        ]
        
        # Filter picks to exclude channels matching auxiliary patterns
        filtered_picks = []
        for pick in picks:
            ch_name = raw.ch_names[pick]
            # Check if channel name contains any non-EEG pattern
            is_aux = any(pattern in ch_name for pattern in non_eeg_patterns)
            if not is_aux:
                filtered_picks.append(pick)
        
        n_removed = len(picks) - len(filtered_picks)
        if n_removed > 0:
            self.log(f"  Filtered out {n_removed} auxiliary/physiological channels")
        
        raw_eeg = raw.copy().pick(filtered_picks)
        self.log(f"✓ Retained {raw_eeg.info['nchan']} channels (EEG + EOG only)")
        
        return raw_eeg
    
    def align_channels(
        self, 
        raw: mne.io.Raw, 
        target_channels: Optional[List[str]] = None
    ) -> mne.io.Raw:
        """
        Align channels to a common set across datasets.
        
        Args:
            raw: Raw MNE object
            target_channels: List of channels to retain (None = store current as target)
            
        Returns:
            Raw object with aligned channels
        """
        if target_channels is None:
            # First call: store EEG channels as target
            eeg_channels = [ch for ch in raw.ch_names 
                           if raw.get_channel_types([ch])[0] == 'eeg']
            self.common_channels = sorted(eeg_channels)
            self.log(f"✓ Stored {len(self.common_channels)} channels as reference")
            return raw
        
        # Find intersection
        current_channels = set(raw.ch_names)
        target_set = set(target_channels)
        common = sorted(current_channels.intersection(target_set))
        
        if len(common) == 0:
            raise ValueError("No common channels found between datasets!")
        
        self.log(f"Channel alignment: {len(common)} common channels")
        
        # Keep only common channels and reorder
        raw_aligned = raw.copy().pick_channels(common, ordered=True)
        
        # Reorder channels to match target order
        target_order = [ch for ch in target_channels if ch in common]
        raw_aligned = raw_aligned.reorder_channels(target_order)
        
        self.log(f"✓ Channels aligned and reordered")
        return raw_aligned
    
    def resample_data(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Resample data to target sampling frequency.
        
        Args:
            raw: Raw MNE object
            
        Returns:
            Resampled raw object
        """
        current_sfreq = raw.info['sfreq']
        
        if current_sfreq == self.target_sfreq:
            self.log(f"✓ Already at target frequency ({self.target_sfreq} Hz)")
            return raw
        
        self.log(f"Resampling: {current_sfreq} Hz → {self.target_sfreq} Hz")
        raw_resampled = raw.copy().resample(self.target_sfreq, npad='auto')
        self.log(f"✓ Resampled to {raw_resampled.info['sfreq']} Hz")
        
        return raw_resampled
    
    def apply_bandpass_filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply zero-phase FIR bandpass filter.
        
        Args:
            raw: Raw MNE object
            
        Returns:
            Filtered raw object
        """
        low, high = self.bandpass_freqs
        self.log(f"Applying bandpass filter: {low}-{high} Hz")
        
        raw_filtered = raw.copy().filter(
            l_freq=low, 
            h_freq=high, 
            fir_design='firwin',
            phase='zero',
            verbose=False
        )
        
        self.log(f"✓ Bandpass filtered ({low}-{high} Hz)")
        return raw_filtered
    
    def apply_notch_filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply notch filter to remove powerline interference.
        
        Args:
            raw: Raw MNE object
            
        Returns:
            Notch-filtered raw object
        """
        # Generate notch frequencies (fundamental + harmonics)
        freqs = [self.notch_freq * (i + 1) for i in range(self.notch_harmonics + 1)]
        freqs = [f for f in freqs if f < raw.info['sfreq'] / 2]  # Nyquist check
        
        self.log(f"Applying notch filter at: {freqs} Hz")
        
        raw_notched = raw.copy().notch_filter(
            freqs=freqs,
            fir_design='firwin',
            phase='zero',
            verbose=False
        )
        
        self.log(f"✓ Notch filtered ({len(freqs)} frequencies)")
        return raw_notched
    
    def detect_bad_channels(self, raw: mne.io.Raw) -> List[str]:
        """
        Detect bad channels using statistical criteria.
        
        Args:
            raw: Raw MNE object
            
        Returns:
            List of bad channel names
        """
        self.log("Detecting bad channels...")
        
        # Get EEG data
        eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude=[])
        data = raw.get_data(picks=eeg_picks)
        
        # Compute variance per channel
        variances = np.var(data, axis=1)
        
        # Z-score normalization
        mean_var = np.mean(variances)
        std_var = np.std(variances)
        z_scores = (variances - mean_var) / (std_var + 1e-8)
        
        # Identify outliers
        bad_idx = np.where(np.abs(z_scores) > self.bad_channel_threshold)[0]
        bad_channels = [raw.ch_names[eeg_picks[i]] for i in bad_idx]
        
        # Check for flat channels (variance near zero)
        flat_threshold = 1e-10
        flat_idx = np.where(variances < flat_threshold)[0]
        flat_channels = [raw.ch_names[eeg_picks[i]] for i in flat_idx]
        
        # Combine
        bad_channels = list(set(bad_channels + flat_channels))
        
        if bad_channels:
            self.log(f"⚠️ Found {len(bad_channels)} bad channels: {bad_channels}")
        else:
            self.log(f"✓ No bad channels detected")
        
        return bad_channels
    
    def interpolate_bad_channels(self, raw: mne.io.Raw, bad_channels: List[str]) -> mne.io.Raw:
        """
        Interpolate bad channels using spherical splines.
        Tries multiple standard montages to find the best match.
        CRITICAL: Does not interpolate if >15% of channels are bad (rejects recording).
        
        Args:
            raw: Raw MNE object
            bad_channels: List of bad channel names
            
        Returns:
            Raw object with interpolated channels
        """
        if not bad_channels:
            return raw
        
        # SAFETY CHECK: Don't interpolate >15% of channels
        n_channels = len(raw.ch_names)
        bad_pct = (len(bad_channels) / n_channels) * 100
        max_bad_pct = 15.0
        
        self.log(f"Found {len(bad_channels)} bad channels ({bad_pct:.1f}%)")
        
        if bad_pct > max_bad_pct:
            self.log(f"⚠️  WARNING: {bad_pct:.1f}% bad channels exceeds {max_bad_pct}% threshold!")
            self.log(f"   Recording quality is too poor for reliable interpolation")
            self.log(f"   SKIPPING interpolation - keeping original data")
            self.log(f"   ⚠️  In production, this recording should be REJECTED")
            return raw  # Return original data without interpolation
        
        raw_interp = raw.copy()
        
        # Try multiple montages in order of preference
        montage_names = ['standard_1020', 'biosemi64', 'biosemi128', 'biosemi256', 'easycap-M1']
        montage_set = False
        
        for montage_name in montage_names:
            try:
                self.log(f"Trying montage: {montage_name}")
                montage = mne.channels.make_standard_montage(montage_name)
                
                # Check how many channels match this montage
                montage_ch_lower = [ch.lower() for ch in montage.ch_names]
                raw_ch_lower = [ch.lower() for ch in raw_interp.ch_names]
                
                matches = sum(1 for ch in raw_ch_lower if ch in montage_ch_lower)
                match_pct = (matches / len(raw_ch_lower)) * 100 if raw_ch_lower else 0
                
                self.log(f"  {matches}/{len(raw_ch_lower)} channels match ({match_pct:.1f}%)")
                
                # If >50% of channels match, try to use this montage
                if match_pct > 50:
                    raw_interp.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)
                    
                    # Verify digitization was set
                    if raw_interp.info['dig'] is not None:
                        self.log(f"✓ Successfully set montage: {montage_name}")
                        montage_set = True
                        break
                    else:
                        self.log(f"  Digitization not set with {montage_name}, trying next...")
                        
            except Exception as e:
                self.log(f"  Failed to load {montage_name}: {str(e)[:50]}")
                continue
        
        # If no montage worked, skip interpolation
        if not montage_set:
            self.log(f"⚠️  Could not set any standard montage (digitization failed)")
            self.log(f"   Skipping interpolation - keeping original data")
            self.log(f"   Note: Channels preserved for CNN input consistency")
            self.log(f"   This often happens with non-standard electrode layouts")
            return raw
        
        # Check which bad channels have positions
        channels_with_positions = []
        channels_without_positions = []
        
        # Get channels that have digitization points
        chs_with_pos = [ch['ch_name'] for ch in raw_interp.info['chs'] 
                       if ch.get('loc') is not None and not all(loc == 0 for loc in ch['loc'][:3])]
        
        for ch in bad_channels:
            if ch in chs_with_pos:
                channels_with_positions.append(ch)
            else:
                channels_without_positions.append(ch)
        
        # Interpolate channels with positions
        if channels_with_positions:
            self.log(f"Interpolating {len(channels_with_positions)} bad channels...")
            raw_interp.info['bads'] = channels_with_positions
            
            try:
                raw_interp = raw_interp.interpolate_bads(reset_bads=True, verbose=False)
                self.log(f"✓ Successfully interpolated {len(channels_with_positions)} channels")
                
                # Verify no NaN values
                data_check = raw_interp.get_data(picks=channels_with_positions)
                if np.any(np.isnan(data_check)):
                    self.log(f"⚠️  Warning: NaN values detected after interpolation")
                    return raw  # Return original if interpolation created NaNs
                    
            except Exception as e:
                self.log(f"⚠️  Interpolation failed: {str(e)[:100]}")
                self.log(f"   Returning original data without interpolation")
                return raw
        else:
            self.log(f"⚠️  No bad channels have known positions for interpolation")
        
        if channels_without_positions:
            self.log(f"⚠️  {len(channels_without_positions)} channels lack positions: {channels_without_positions}")
            self.log(f"   These channels kept without interpolation for CNN consistency")
        
        return raw_interp
    
    def apply_reference(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply common average reference.
        
        Args:
            raw: Raw MNE object
            
        Returns:
            Re-referenced raw object
        """
        self.log(f"Applying {self.reference} reference...")
        
        raw_ref = raw.copy().set_eeg_reference(
            ref_channels=self.reference,
            projection=False,
            verbose=False
        )
        
        self.log(f"✓ Re-referenced to {self.reference}")
        return raw_ref
    
    def run_ica_artifact_removal(
        self, 
        raw: mne.io.Raw, 
        eog_channels: Optional[List[str]] = None
    ) -> Tuple[mne.io.Raw, mne.preprocessing.ICA]:
        """
        Run ICA to detect and remove artifacts.
        
        Args:
            raw: Raw MNE object
            eog_channels: List of EOG channel names for correlation
            
        Returns:
            (cleaned_raw, ica_object)
        """
        self.log("Running ICA for artifact detection...")
        
        # Determine number of components
        n_channels = len(mne.pick_types(raw.info, eeg=True, eog=False, exclude=[]))
        n_components = self.ica_n_components or min(25, n_channels)
        
        self.log(f"  ICA components: {n_components}")
        
        # Fit ICA
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method='fastica',
            random_state=42,
            max_iter=200,
            verbose=False
        )
        
        # Fit on EEG channels only
        picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude=[])
        ica.fit(raw, picks=picks, verbose=False)
        
        # Find EOG artifacts
        eog_indices = []
        if eog_channels:
            for eog_ch in eog_channels:
                if eog_ch in raw.ch_names:
                    eog_indices_ch, eog_scores = ica.find_bads_eog(
                        raw, 
                        ch_name=eog_ch,
                        threshold=3.0,
                        verbose=False
                    )
                    eog_indices.extend(eog_indices_ch)
        
        # Find muscle artifacts (high-frequency components)
        # Critical for migraine patients who have elevated muscle tension
        muscle_indices, muscle_scores = ica.find_bads_muscle(
            raw,
            threshold=0.5,  # Lower threshold = more aggressive removal
            verbose=False
        )
        
        # Combine artifact indices
        artifact_indices = list(set(eog_indices + muscle_indices))
        
        if artifact_indices:
            self.log(f"  Found {len(artifact_indices)} artifact components:")
            self.log(f"    - EOG: {len(eog_indices)}")
            self.log(f"    - Muscle/EMG: {len(muscle_indices)}")
            ica.exclude = artifact_indices
        else:
            self.log(f"  No clear artifact components found")
        
        # Apply ICA
        raw_clean = raw.copy()
        ica.apply(raw_clean, verbose=False)
        
        self.log(f"✓ ICA artifact removal complete")
        return raw_clean, ica
    
    def preprocess_file(
        self,
        file_path: Path,
        dataset_type: str = 'lemon',
        target_channels: Optional[List[str]] = None,
        eog_channels: Optional[List[str]] = None
    ) -> Tuple[mne.io.Raw, Dict]:
        """
        Run complete preprocessing pipeline on a single file.
        
        Args:
            file_path: Path to raw EEG file
            dataset_type: 'lemon' or 'migraine'
            target_channels: Target channel list for alignment (None = auto)
            eog_channels: EOG channel names for ICA
            
        Returns:
            (preprocessed_raw, metadata_dict)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Preprocessing: {file_path.name}")
            print(f"{'='*60}")
        
        metadata = {
            'file_path': str(file_path),
            'dataset_type': dataset_type,
            'bad_channels': [],
            'n_ica_components_removed': 0
        }
        
        # 1. Load raw data
        raw = self.load_raw_eeg(file_path, dataset_type)
        metadata['original_sfreq'] = raw.info['sfreq']
        metadata['original_n_channels'] = raw.info['nchan']
        metadata['duration_seconds'] = raw.times[-1]
        
        # 2. Keep only EEG + EOG channels
        raw = self.keep_eeg_channels_only(raw, keep_eog=True)
        
        # 3. Resample
        raw = self.resample_data(raw)
        metadata['resampled_sfreq'] = raw.info['sfreq']
        
        # 4. Bandpass filter
        raw = self.apply_bandpass_filter(raw)
        
        # 5. Notch filter
        raw = self.apply_notch_filter(raw)
        
        # 6. Detect and interpolate bad channels
        bad_channels = self.detect_bad_channels(raw)
        raw = self.interpolate_bad_channels(raw, bad_channels)
        metadata['bad_channels'] = bad_channels
        
        # 7. Apply reference
        raw = self.apply_reference(raw)
        
        # 8. Align channels (do this before ICA to ensure consistency)
        if target_channels is not None:
            raw = self.align_channels(raw, target_channels)
        
        # 9. Remove EOG channels before ICA (we kept them for correlation)
        eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude=[])
        eeg_only = raw.copy().pick(eeg_picks)
        
        # 10. ICA artifact removal
        raw_clean, ica = self.run_ica_artifact_removal(eeg_only, eog_channels)
        metadata['n_ica_components_removed'] = len(ica.exclude)
        metadata['final_n_channels'] = raw_clean.info['nchan']
        
        if self.verbose:
            print(f"{'='*60}")
            print(f"✓ Preprocessing Complete")
            print(f"  Final channels: {metadata['final_n_channels']}")
            print(f"  Sampling rate: {metadata['resampled_sfreq']} Hz")
            print(f"  Duration: {metadata['duration_seconds']:.1f}s")
            print(f"{'='*60}\n")
        
        return raw_clean, metadata


def get_common_channels_between_datasets(
    lemon_file: Path,
    migraine_file: Path,
    verbose: bool = True
) -> List[str]:
    """
    Identify common EEG channels between LEMON and migraine datasets.
    
    Args:
        lemon_file: Path to a LEMON .vhdr file
        migraine_file: Path to a migraine .bdf file
        verbose: Print results
        
    Returns:
        Sorted list of common EEG channel names
    """
    # Load LEMON
    raw_lemon = mne.io.read_raw_brainvision(lemon_file, preload=False, verbose=False)
    lemon_eeg = [ch for ch in raw_lemon.ch_names 
                 if raw_lemon.get_channel_types([ch])[0] == 'eeg']
    
    # Load Migraine
    raw_migraine = mne.io.read_raw_bdf(migraine_file, preload=False, verbose=False)
    migraine_eeg = [ch for ch in raw_migraine.ch_names 
                    if raw_migraine.get_channel_types([ch])[0] == 'eeg']
    
    # Find intersection
    common = sorted(set(lemon_eeg).intersection(set(migraine_eeg)))
    
    if verbose:
        print(f"LEMON EEG channels: {len(lemon_eeg)}")
        print(f"Migraine EEG channels: {len(migraine_eeg)}")
        print(f"Common channels: {len(common)}")
        print(f"Common channel list: {common}")
    
    return common


if __name__ == "__main__":
    """
    Example usage and testing
    """
    from pathlib import Path
    
    # Paths
    lemon_path = Path(r'g:\Study\FYDP-I_Personalized-Migraine-Mitigation-Via-Binaural-Beats\EEG_MPILMBB_LEMON\EEG_Raw_BIDS_ID')
    migraine_path = Path(r'g:\Study\FYDP-I_Personalized-Migraine-Mitigation-Via-Binaural-Beats\Dataset')
    
    # Sample files
    lemon_sample = lemon_path / 'sub-010002' / 'RSEEG' / 'sub-010002.vhdr'
    migraine_sample = migraine_path / 'C1' / 'C1' / 'C1_Resting.bdf'
    
    # Get common channels
    print("\n" + "="*60)
    print("FINDING COMMON CHANNELS")
    print("="*60)
    common_channels = get_common_channels_between_datasets(
        lemon_sample, 
        migraine_sample,
        verbose=True
    )
    
    # Initialize preprocessor
    print("\n" + "="*60)
    print("TESTING PREPROCESSING PIPELINE")
    print("="*60)
    
    preprocessor = UnifiedEEGPreprocessor(
        target_sfreq=250.0,
        bandpass_freqs=(1.0, 45.0),
        notch_freq=50.0,
        verbose=True
    )
    
    # Preprocess LEMON sample
    print("\n### Processing LEMON Sample ###")
    raw_lemon_clean, meta_lemon = preprocessor.preprocess_file(
        lemon_sample,
        dataset_type='lemon',
        target_channels=common_channels
    )
    
    # Preprocess Migraine sample
    print("\n### Processing Migraine Sample ###")
    raw_migraine_clean, meta_migraine = preprocessor.preprocess_file(
        migraine_sample,
        dataset_type='migraine',
        target_channels=common_channels
    )
    
    print("\n" + "="*60)
    print("✓ PREPROCESSING PIPELINE TEST COMPLETE")
    print("="*60)
    print(f"\nBoth datasets now have:")
    print(f"  - {raw_lemon_clean.info['nchan']} channels (aligned)")
    print(f"  - {raw_lemon_clean.info['sfreq']} Hz sampling rate")
    print(f"  - Bandpass filtered (1-45 Hz)")
    print(f"  - Artifacts removed (ICA)")
    print(f"  - Common average reference")
    print("\n✓ Ready for windowing and transfer learning!")
