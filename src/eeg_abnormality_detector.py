"""
Precise EEG Abnormality Detector
Channel-level statistical analysis with topographic mapping
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle
from pathlib import Path
import sys
sys.path.insert(0, '/Users/mahmudulmashrafe/Programming/FYDP/3/src')


def build_control_database(control_patients, save_path='data/control_database.pkl'):
    """
    Build statistical database from control subjects
    
    Args:
        control_patients: List of control patient IDs
        save_path: Where to save the database
        
    Returns:
        Database with mean and std for each channel and frequency band
    """
    from data_loader import load_all_tasks
    from advanced_features import extract_fine_frequency_bands
    
    print(f"Building control database from {len(control_patients)} patients...")
    
    # Initialize storage
    all_band_powers = {}
    
    for patient_id in control_patients:
        print(f"  Processing {patient_id}...")
        tasks = load_all_tasks(patient_id, verbose=False)
        
        if tasks['resting'] is not None:
            band_powers = extract_fine_frequency_bands(tasks['resting'], bands_config='fine')
            
            for band_name, powers in band_powers.items():
                if band_name not in all_band_powers:
                    all_band_powers[band_name] = []
                all_band_powers[band_name].append(powers)
    
    # Calculate statistics
    database = {}
    for band_name, powers_list in all_band_powers.items():
        powers_array = np.array(powers_list)  #  (n_controls, n_channels)
        database[band_name] = {
            'mean': np.mean(powers_array, axis=0),
            'std': np.std(powers_array, axis=0),
            'n': len(powers_list)
        }
    
    # Save database
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    with open(save_path, 'wb') as f:
        pickle.dump(database, f)
    
    print(f"✓ Control database saved to {save_path}")
    return database


def load_control_database(path='data/control_database.pkl'):
    """Load pre-built control database"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def detect_channel_abnormalities(patient_id, control_database, significance_level=0.05):
    """
    Detect channel-specific abnormalities with statistical testing
    
    Args:
        patient_id: Patient to analyze
        control_database: Database from build_control_database()
        significance_level: P-value threshold (default 0.05)
        
    Returns:
        Dictionary with abnormalities per channel and band
    """
    from data_loader import load_all_tasks
    from advanced_features import extract_fine_frequency_bands
    
    # Load patient data
    tasks = load_all_tasks(patient_id, verbose=False)
    
    if tasks['resting'] is None:
        raise ValueError(f"No resting state data for {patient_id}")
    
    # Extract patient's band powers
    patient_powers = extract_fine_frequency_bands(tasks['resting'], bands_config='fine')
    
    # Detect abnormalities
    abnormalities = []
    
    for band_name, powers in patient_powers.items():
        if band_name not in control_database:
            continue
        
        control_mean = control_database[band_name]['mean']
        control_std = control_database[band_name]['std']
        
        # Calculate z-scores for each channel
        z_scores = (powers - control_mean) / (control_std + 1e-10)
        
        # Calculate p-values (two-tailed test)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        
        # Find significant abnormalities
        significant_idx = p_values < significance_level
        
        if np.any(significant_idx):
            for ch_idx in np.where(significant_idx)[0]:
                abnormalities.append({
                    'band': band_name,
                    'channel_idx': ch_idx,
                    'patient_power': powers[ch_idx],
                    'control_mean': control_mean[ch_idx],
                    'control_std': control_std[ch_idx],
                    'z_score': z_scores[ch_idx],
                    'p_value': p_values[ch_idx],
                    'severity': 'severe' if np.abs(z_scores[ch_idx]) > 3 else ('moderate' if np.abs(z_scores[ch_idx]) > 2 else 'mild'),
                    'direction': 'excess' if z_scores[ch_idx] > 0 else 'deficit'
                })
    
    # Sort by significance
    abnormalities = sorted(abnormalities, key=lambda x: x['p_value'])
    
    return abnormalities


def identify_regional_abnormalities(abnormalities, channel_names=None):
    """
    Group channel abnormalities into brain regions
    
    Args:
        abnormalities: List from detect_channel_abnormalities()
        channel_names: List of channel names (optional)
        
    Returns:
        Dictionary with regional summary
    """
    # Define brain regions based on channel indices
    # This is a simplified mapping - ideally use actual channel names
    regions = {
        'frontal': list(range(0, 32)),
        'central': list(range(32, 64)),
        'parietal': list(range(64, 96)),
        'occipital': list(range(96, 128))
    }
    
    regional_summary = {}
    
    for region_name, channel_indices in regions.items():
        region_abnormalities = [a for a in abnormalities if a['channel_idx'] in channel_indices]
        
        if region_abnormalities:
            # Group by band
            bands_affected = {}
            for abn in region_abnormalities:
                band = abn['band']
                if band not in bands_affected:
                    bands_affected[band] = []
                bands_affected[band].append(abn)
            
            regional_summary[region_name] = {
                'n_channels_affected': len(set(a['channel_idx'] for a in region_abnormalities)),
                'bands_affected': bands_affected,
                'primary_abnormality': region_abnormalities[0]  # Most significant
            }
    
    return regional_summary


def calculate_abnormality_score(abnormalities):
    """
    Calculate overall abnormality severity score
    
    Args:
        abnormalities: List from detect_channel_abnormalities()
        
    Returns:
        Score from 0 (normal) to 10 (severe)
    """
    if not abnormalities:
        return 0.0
    
    # Weight by severity
    severity_weights = {'mild': 1, 'moderate': 2, 'severe': 3}
    
    total_score = sum(severity_weights[a['severity']] * np.abs(a['z_score']) for a in abnormalities)
    
    # Normalize to 0-10 scale
    score = min(10.0, total_score / len(abnormalities))
    
    return score


def generate_abnormality_report(patient_id, abnormalities, regional_summary, score):
    """
    Generate detailed text report of abnormalities
    
    Args:
        patient_id: Patient identifier
        abnormalities: From detect_channel_abnormalities()
        regional_summary: From identify_regional_abnormalities()
        score: From calculate_abnormality_score()
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 70)
    report.append(f"PRECISE EEG ABNORMALITY REPORT - {patient_id}")
    report.append("=" * 70)
    report.append("")
    
    # Overall score
    report.append(f"OVERALL ABNORMALITY SCORE: {score:.1f}/10")
    if score < 3:
        report.append("Interpretation: MILD abnormalities")
    elif score < 6:
        report.append("Interpretation: MODERATE abnormalities")
    else:
        report.append("Interpretation: SEVERE abnormalities")
    report.append("")
    report.append("-" * 70)
    report.append("")
    
    # Regional summary
    report.append("REGIONAL ABNORMALITIES:")
    report.append("")
    
    if not regional_summary:
        report.append("  No significant regional abnormalities detected")
    else:
        for region, data in regional_summary.items():
            report.append(f"  {region.upper()} REGION:")
            report.append(f"    • Affected channels: {data['n_channels_affected']}")
            report.append(f"    • Frequency bands:")
            
            for band, abn_list in data['bands_affected'].items():
                avg_z = np.mean([a['z_score'] for a in abn_list])
                direction = "EXCESS" if avg_z > 0 else "DEFICIT"
                report.append(f"      - {band}: {direction} (z={avg_z:.2f}, n={len(abn_list)} channels)")
            
            # Primary abnormality
            primary = data['primary_abnormality']
            report.append(f"    • Most significant: {primary['band']}, channel {primary['channel_idx']}")
            report.append(f"      z={primary['z_score']:.2f}, p={primary['p_value']:.4f}")
            report.append("")
    
    report.append("-" * 70)
    report.append("")
    
    # Top abnormalities
    report.append("TOP 10 MOST SIGNIFICANT ABNORMALITIES:")
    report.append("")
    
    for i, abn in enumerate(abnormalities[:10], 1):
        report.append(f"  {i}. Channel {abn['channel_idx']} - {abn['band']}")
        report.append(f"     Direction: {abn['direction'].upper()}")
        report.append(f"     Severity: {abn['severity'].upper()}")
        report.append(f"     Patient: {abn['patient_power']:.2e}, Control: {abn['control_mean']:.2e}")
        report.append(f"     Z-score: {abn['z_score']:.2f}, P-value: {abn['p_value']:.4f}")
        report.append("")
    
    report.append("=" * 70)
    
    return "\n".join(report)


if __name__ == "__main__":
    """Test abnormality detection"""
    print("=" * 70)
    print("Testing Precise EEG Abnormality Detector")
    print("=" * 70)
    
    # First, build control database
    from data_loader import get_all_patient_ids, load_clinical_data
    
    clinical_df = load_clinical_data()
    all_patients = get_all_patient_ids()
    
    # Get control patients
    control_patients = [p for p in all_patients if p.startswith('C')][:10]  # Use first 10 for testing
    
    print(f"\nStep 1: Building control database from {len(control_patients)} patients...")
    control_db = build_control_database(control_patients)
    
    # Test on a migraine patient
    test_patient = 'M1_1'
    print(f"\nStep 2: Detecting abnormalities in {test_patient}...")
    abnormalities = detect_channel_abnormalities(test_patient, control_db)
    
    print(f"  Found {len(abnormalities)} significant abnormalities")
    
    # Regional analysis
    print(f"\nStep 3: Identifying regional patterns...")
    regional = identify_regional_abnormalities(abnormalities)
    
    # Calculate score
    score = calculate_abnormality_score(abnormalities)
    print(f"  Abnormality score: {score:.1f}/10")
    
    # Generate report
    print(f"\nStep 4: Generating report...")
    report = generate_abnormality_report(test_patient, abnormalities, regional, score)
    print("\n" + report)
    
    print("\n" + "=" * 70)
    print("Abnormality Detection Test Complete!")
    print("=" * 70)
