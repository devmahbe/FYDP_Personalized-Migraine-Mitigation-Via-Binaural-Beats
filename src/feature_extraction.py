"""
EEG Feature Extraction Module
Extracts meaningful features from raw EEG signals
"""
import numpy as np
import mne
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


def extract_psd_features(raw, fmin=0.5, fmax=50):
    """
    Extract Power Spectral Density features from EEG
    
    Frequency bands:
    - Delta: 0.5-4 Hz
    - Theta: 4-8 Hz
    - Alpha: 8-13 Hz
    - Beta: 13-30 Hz
    - Gamma: 30-50 Hz
    
    Args:
        raw: MNE Raw object
        fmin, fmax: Frequency range for analysis
        
    Returns:
        numpy array of PSD features (5 bands × num_channels)
    """
    # Get data
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # Only use EEG channels (exclude EOG, etc.)
    ch_types = raw.get_channel_types()
    eeg_indices = [i for i, ch_type in enumerate(ch_types) if ch_type == 'eeg']
    
    if len(eeg_indices) == 0:
        # If no EEG channels marked, assume first 128 are EEG
        eeg_indices = list(range(min(128, data.shape[0])))
    
    eeg_data = data[eeg_indices, :]
    
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    features = []
    
    for ch_idx in range(eeg_data.shape[0]):
        # Compute PSD using Welch's method
        freqs, psd = signal.welch(eeg_data[ch_idx, :], sfreq, nperseg=min(2048, eeg_data.shape[1]))
        
        # Extract power in each band
        for band_name, (low, high) in bands.items():
            # Find frequencies in this band
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            # Average power in band
            band_power = np.mean(psd[idx_band])
            features.append(band_power)
    
    return np.array(features)


def extract_statistical_features(raw):
    """
    Extract statistical features from EEG time series
    
    Features per channel:
    - Mean
    - Variance
    - Skewness
    - Kurtosis
    
    Args:
        raw: MNE Raw object
        
    Returns:
        numpy array of statistical features (4 × num_channels)
    """
    # Get data
    data = raw.get_data()
    
    # Only use EEG channels
    ch_types = raw.get_channel_types()
    eeg_indices = [i for i, ch_type in enumerate(ch_types) if ch_type == 'eeg']
    
    if len(eeg_indices) == 0:
        eeg_indices = list(range(min(128, data.shape[0])))
    
    eeg_data = data[eeg_indices, :]
    
    features = []
    
    for ch_idx in range(eeg_data.shape[0]):
        ch_data = eeg_data[ch_idx, :]
        
        # Statistical features
        features.append(np.mean(ch_data))
        features.append(np.var(ch_data))
        features.append(skew(ch_data))
        features.append(kurtosis(ch_data))
    
    return np.array(features)


def extract_connectivity_features(raw, n_pairs=20):
    """
    Extract connectivity features (coherence) between electrode pairs
    
    Args:
        raw: MNE Raw object
        n_pairs: Number of electrode pairs to compute (randomly sampled)
        
    Returns:
        numpy array of coherence values
    """
    # Get data
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # Only use EEG channels
    ch_types = raw.get_channel_types()
    eeg_indices = [i for i, ch_type in enumerate(ch_types) if ch_type == 'eeg']
    
    if len(eeg_indices) == 0:
        eeg_indices = list(range(min(128, data.shape[0])))
    
    eeg_data = data[eeg_indices, :]
    n_channels = eeg_data.shape[0]
    
    # Randomly select electrode pairs (for computational efficiency)
    np.random.seed(42)  # For reproducibility
    pairs = []
    for _ in range(min(n_pairs, n_channels)):
        i, j = np.random.choice(n_channels, size=2, replace=False)
        pairs.append((i, j))
    
    features = []
    
    for i, j in pairs:
        # Compute coherence in alpha band (8-13 Hz)
        freqs, coh = signal.coherence(eeg_data[i, :], eeg_data[j, :], sfreq, nperseg=min(1024, eeg_data.shape[1]))
        
        # Average coherence in alpha band
        alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
        alpha_coh = np.mean(coh[alpha_idx])
        features.append(alpha_coh)
    
    return np.array(features)


def extract_band_ratios(raw):
    """
    Extract band power ratios (useful for migraine detection)
    
    Ratios:
    - Theta/Alpha ratio
    - Delta/Alpha ratio
    - (Theta+Alpha)/(Beta+Gamma) ratio
    
    Args:
        raw: MNE Raw object
        
    Returns:
        numpy array of ratio features (3 × num_channels)
    """
    # Get data
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # Only use EEG channels
    ch_types = raw.get_channel_types()
    eeg_indices = [i for i, ch_type in enumerate(ch_types) if ch_type == 'eeg']
    
    if len(eeg_indices) == 0:
        eeg_indices = list(range(min(128, data.shape[0])))
    
    eeg_data = data[eeg_indices, :]
    
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    features = []
    
    for ch_idx in range(eeg_data.shape[0]):
        # Compute PSD
        freqs, psd = signal.welch(eeg_data[ch_idx, :], sfreq, nperseg=min(2048, eeg_data.shape[1]))
        
        # Extract power in each band
        band_powers = {}
        for band_name, (low, high) in bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            band_powers[band_name] = np.mean(psd[idx_band])
        
        # Compute ratios (add small epsilon to avoid division by zero)
        eps = 1e-10
        theta_alpha_ratio = band_powers['theta'] / (band_powers['alpha'] + eps)
        delta_alpha_ratio = band_powers['delta'] / (band_powers['alpha'] + eps)
        slow_fast_ratio = (band_powers['theta'] + band_powers['alpha']) / (band_powers['beta'] + band_powers['gamma'] + eps)
        
        features.extend([theta_alpha_ratio, delta_alpha_ratio, slow_fast_ratio])
    
    return np.array(features)


def extract_all_features(patient_id, task='resting', verbose=False):
    """
    Extract all features for a patient
    
    Args:
        patient_id: Patient identifier
        task: EEG task type ('resting', 'SSAEP', 'SSVEP')
        verbose: Print progress
        
    Returns:
        numpy array of all features combined
    """
    from data_loader import load_eeg_file
    
    if verbose:
        print(f"Loading {patient_id} - {task}...")
    
    # Load EEG data
    try:
        raw = load_eeg_file(patient_id, task, verbose=False)
    except FileNotFoundError:
        if verbose:
            print(f"  ✗ File not found for {patient_id} - {task}")
        return None
    
    # Extract features
    if verbose:
        print(f"  Extracting PSD features...")
    psd_features = extract_psd_features(raw)
    
    if verbose:
        print(f"  Extracting statistical features...")
    stat_features = extract_statistical_features(raw)
    
    if verbose:
        print(f"  Extracting connectivity features...")
    conn_features = extract_connectivity_features(raw, n_pairs=20)
    
    if verbose:
        print(f"  Extracting band ratios...")
    ratio_features = extract_band_ratios(raw)
    
    # Combine all features
    all_features = np.concatenate([psd_features, stat_features, conn_features, ratio_features])
    
    if verbose:
        print(f"  ✓ Total features: {len(all_features)}")
    
    return all_features


if __name__ == "__main__":
    """Test feature extraction"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    print("=" * 60)
    print("Testing Feature Extraction Module")
    print("=" * 60)
    
    # Test on M1_1
    print("\nExtracting features for M1_1 (resting state)...")
    features = extract_all_features('M1_1', task='resting', verbose=True)
    
    if features is not None:
        print(f"\n✓ Successfully extracted {len(features)} features")
        print(f"  Feature vector shape: {features.shape}")
        print(f"  First 10 features: {features[:10]}")
        print(f"  Feature statistics:")
        print(f"    - Min: {np.min(features):.6f}")
        print(f"    - Max: {np.max(features):.6f}")
        print(f"    - Mean: {np.mean(features):.6f}")
        print(f"    - Std: {np.std(features):.6f}")
    
    # Test on a control patient
    print("\n" + "-" * 60)
    print("\nExtracting features for C1 (resting state)...")
    features_c1 = extract_all_features('C1', task='resting', verbose=True)
    
    if features_c1 is not None:
        print(f"\n✓ Successfully extracted {len(features_c1)} features")
        print(f"  Feature vector shape: {features_c1.shape}")
    
    print("\n" + "=" * 60)
    print("Feature Extraction Test Complete!")
    print("=" * 60)
