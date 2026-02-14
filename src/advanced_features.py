"""
Advanced Feature Extraction Module
Implements fine-grained frequency analysis and advanced EEG features
"""
import numpy as np
import mne
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def extract_fine_frequency_bands(raw, bands_config='fine'):
    """
    Extract power in fine-grained frequency sub-bands
    
    Args:
        raw: MNE Raw object
        bands_config: 'fine' for 15 sub-bands, 'coarse' for 5 bands
        
    Returns:
        Dictionary with band powers per channel
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
    
    # Define fine-grained frequency bands
    if bands_config == 'fine':
        bands = {
            # Delta sub-bands
            'delta_low': (0.5, 1.0),
            'delta_mid': (1.0, 2.0),
            'delta_high': (2.0, 4.0),
            # Theta sub-bands
            'theta_low': (4.0, 6.0),
            'theta_high': (6.0, 8.0),
            # Alpha sub-bands (critical for migraine)
            'alpha_low': (8.0, 10.0),   # Reduced in migraine
            'alpha_mid': (10.0, 12.0),
            'alpha_high': (12.0, 13.0),
            # Beta sub-bands
            'beta_low': (13.0, 20.0),
            'beta_high': (20.0, 30.0),
            # Gamma sub-bands
            'gamma_low': (30.0, 40.0),
            'gamma_high': (40.0, 50.0),
            # Additional sub-bands
            'theta_alpha': (7.0, 9.0),  # Transition zone
            'mu': (8.0, 12.0),          # Sensorimotor rhythm
            'slow_wave': (0.5, 2.0),    # Very slow activity
        }
    else:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    # Extract power for each band and channel
    band_powers = {band_name: np.zeros(n_channels) for band_name in bands.keys()}
    
    for ch_idx in range(n_channels):
        # Compute PSD using Welch's method
        freqs, psd = signal.welch(eeg_data[ch_idx, :], sfreq, nperseg=min(2048, eeg_data.shape[1]))
        
        for band_name, (low, high) in bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            band_powers[band_name][ch_idx] = np.mean(psd[idx_band])
    
    return band_powers


def extract_functional_connectivity(raw, method='coherence', n_pairs=50):
    """
    Extract functional connectivity features using coherence
    
    Args:
        raw: MNE Raw object
        method: 'coherence' or 'correlation'
        n_pairs: Number of channel pairs to analyze
        
    Returns:
        Connectivity features (mean coherence per frequency band)
    """
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # Only use EEG channels
    ch_types = raw.get_channel_types()
    eeg_indices = [i for i, ch_type in enumerate(ch_types) if ch_type == 'eeg']
    
    if len(eeg_indices) == 0:
        eeg_indices = list(range(min(128, data.shape[0])))
    
    eeg_data = data[eeg_indices, :]
    n_channels = eeg_data.shape[0]
    
    # Select strategically important channel pairs
    # Focus on: frontal-occipital, left-right, local connections
    pairs = []
    
    # Long-range connections (frontal-occipital)
    if n_channels >= 8:
        pairs.extend([(0, n_channels-1), (1, n_channels-2)])
    
    # Interhemispheric (left-right)
    if n_channels >= 64:
        pairs.extend([(i, i + n_channels//2) for i in range(min(5, n_channels//2))])
    
    # Random pairs for diversity
    np.random.seed(42)
    remaining = n_pairs - len(pairs)
    for _ in range(remaining):
        i, j = np.random.choice(n_channels, size=2, replace=False)
        pairs.append((i, j))
    
    # Compute coherence for each pair
    connectivity_features = []
    
    for i, j in pairs[:n_pairs]:
        freqs, coh = signal.coherence(eeg_data[i, :], eeg_data[j, :], sfreq, 
                                      nperseg=min(1024, eeg_data.shape[1]))
        
        # Average coherence in each major band
        # Delta
        delta_idx = np.logical_and(freqs >= 0.5, freqs <= 4)
        connectivity_features.append(np.mean(coh[delta_idx]))
        
        # Theta
        theta_idx = np.logical_and(freqs >= 4, freqs <= 8)
        connectivity_features.append(np.mean(coh[theta_idx]))
        
        # Alpha (critical)
        alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
        connectivity_features.append(np.mean(coh[alpha_idx]))
        
        # Beta
        beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
        connectivity_features.append(np.mean(coh[beta_idx]))
    
    return np.array(connectivity_features)


def extract_nonlinear_features(raw, sample_length=1000):
    """
    Extract nonlinear dynamics features
    
    Args:
        raw: MNE Raw object
        sample_length: Number of samples for entropy calculation
        
    Returns:
        Nonlinear features (entropy, complexity)
    """
    data = raw.get_data()
    
    # Only use EEG channels
    ch_types = raw.get_channel_types()
    eeg_indices = [i for i, ch_type in enumerate(ch_types) if ch_type == 'eeg']
    
    if len(eeg_indices) == 0:
        eeg_indices = list(range(min(128, data.shape[0])))
    
    eeg_data = data[eeg_indices, :]
    n_channels = eeg_data.shape[0]
    
    features = []
    
    # Sample entropy for each channel
    for ch_idx in range(n_channels):
        # Use a subsample for efficiency
        ch_data = eeg_data[ch_idx, :min(sample_length, eeg_data.shape[1])]
        
        # Approximate entropy
        try:
            ent = approximate_entropy(ch_data, m=2, r=0.2*np.std(ch_data))
            features.append(ent)
        except:
            features.append(0)
        
        # Hjorth parameters (activity, mobility, complexity)
        activity = np.var(ch_data)
        features.append(activity)
        
        # Mobility
        diff1 = np.diff(ch_data)
        mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
        features.append(mobility)
        
        # Complexity
        diff2 = np.diff(diff1)
        complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
        features.append(complexity)
    
    return np.array(features)


def approximate_entropy(signal, m=2, r=0.2):
    """
    Calculate approximate entropy of a signal
    
    Args:
        signal: 1D array
        m: Pattern length
        r: Tolerance (fraction of std)
        
    Returns:
        Approximate entropy value
    """
    N = len(signal)
    
    def _maxdist(xi, xj):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def _phi(m):
        patterns = np.array([[signal[j] for j in range(i, i + m)] for i in range(N - m + 1)])
        C = np.zeros(N - m + 1)
        for i in range(N - m + 1):
            for j in range(N - m + 1):
                if _maxdist(patterns[i], patterns[j]) <= r:
                    C[i] += 1
        C = C / (N - m + 1.0)
        return np.sum(np.log(C + 1e-10)) / (N - m + 1.0)
    
    return _phi(m) - _phi(m + 1)


def extract_advanced_all_features(patient_id, tasks_data, verbose=False):
    """
    Extract advanced features from multi-task EEG data
    
    Args:
        patient_id: Patient identifier
        tasks_data: Dictionary with {'resting': raw, 'SSAEP': raw, 'SSVEP': raw}
        verbose: Print progress
        
    Returns:
        Comprehensive feature vector
    """
    all_features = []
    
    # Process each available task
    for task_name, raw in tasks_data.items():
        if raw is None:
            if verbose:
                print(f"  Skipping {task_name} (not available)")
            continue
        
        if verbose:
            print(f"  Processing {task_name}...")
        
        # 1. Fine-grained frequency bands
        band_powers = extract_fine_frequency_bands(raw, bands_config='fine')
        for band_name, powers in band_powers.items():
            all_features.extend(powers)
        
        if verbose:
            print(f"    - Fine-grained bands: {sum(len(p) for p in band_powers.values())} features")
        
        # 2. Statistical features (only for resting to avoid redundancy)
        if task_name == 'resting':
            from feature_extraction import extract_statistical_features
            stat_features = extract_statistical_features(raw)
            all_features.extend(stat_features)
            
            if verbose:
                print(f"    - Statistical: {len(stat_features)} features")
        
        # 3. Functional connectivity
        conn_features = extract_functional_connectivity(raw, n_pairs=20)
        all_features.extend(conn_features)
        
        if verbose:
            print(f"    - Connectivity: {len(conn_features)} features")
        
        # 4. Nonlinear features (only for resting to avoid redundancy)
        if task_name == 'resting':
            nonlinear_features = extract_nonlinear_features(raw, sample_length=1000)
            all_features.extend(nonlinear_features)
            
            if verbose:
                print(f"    - Nonlinear: {len(nonlinear_features)} features")
    
    return np.array(all_features)


if __name__ == "__main__":
    """Test advanced features"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from data_loader import load_all_tasks
    
    print("=" * 70)
    print("Testing Advanced Feature Extraction")
    print("=" * 70)
    
    # Load multi-task data
    patient_id = 'M1_1'
    print(f"\nLoading all tasks for {patient_id}...")
    tasks_data = load_all_tasks(patient_id, verbose=True)
    
    # Extract advanced features
    print(f"\nExtracting advanced features...")
    features = extract_advanced_all_features(patient_id, tasks_data, verbose=True)
    
    print(f"\n✓ Total advanced features: {len(features)}")
    print(f"  Feature statistics:")
    print(f"    - Min: {np.nanmin(features):.6e}")
    print(f"    - Max: {np.nanmax(features):.6e}")
    print(f"    - Mean: {np.nanmean(features):.6e}")
    print(f"    - NaN count: {np.sum(np.isnan(features))}")
    
    print("\n" + "=" * 70)
    print("Advanced Feature Extraction Test Complete!")
    print("=" * 70)
