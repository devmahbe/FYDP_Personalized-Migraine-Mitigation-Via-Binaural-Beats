"""
Binaural Beat Generator Module
Generate personalized therapeutic audio for migraine patients
"""
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path


def analyze_eeg_abnormality(features):
    """
    Analyze EEG features to detect frequency band abnormalities
    
    Args:
        features: Feature vector from feature_extraction
        
    Returns:
        Dictionary with band power information
    """
    # Assuming first 640 features are PSD (5 bands × 128 channels)
    # Each band has 128 values
    band_size = 128
    
    try:
        delta_power = np.mean(features[0:band_size])
        theta_power = np.mean(features[band_size:2*band_size])
        alpha_power = np.mean(features[2*band_size:3*band_size])
        beta_power = np.mean(features[3*band_size:4*band_size])
        gamma_power = np.mean(features[4*band_size:5*band_size])
        
        # Calculate ratios
        total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
        if total_power > 0:
            delta_ratio = delta_power / total_power
            theta_ratio = theta_power / total_power
            alpha_ratio = alpha_power / total_power
            beta_ratio = beta_power / total_power
        else:
            delta_ratio = theta_ratio = alpha_ratio = beta_ratio = 0.25
        
        abnormality = {
            'delta_power': delta_power,
            'theta_power': theta_power,
            'alpha_power': alpha_power,
            'beta_power': beta_power,
            'gamma_power': gamma_power,
            'delta_ratio': delta_ratio,
            'theta_ratio': theta_ratio,
            'alpha_ratio': alpha_ratio,
            'beta_ratio': beta_ratio,
            'dominant_band': None
        }
        
        # Determine dominant abnormal band
        if delta_ratio > 0.35:
            abnormality['dominant_band'] = 'delta'
        elif theta_ratio > 0.30:
            abnormality['dominant_band'] = 'theta'
        elif beta_ratio > 0.30:
            abnormality['dominant_band'] = 'beta'
        else:
            abnormality['dominant_band'] = 'normal'
        
        return abnormality
    
    except Exception as e:
        # Fallback if feature structure is different
        return {
            'dominant_band': 'normal',
            'delta_ratio': 0.25,
            'theta_ratio': 0.25,
            'alpha_ratio': 0.25,
            'beta_ratio': 0.25
        }


def calculate_optimal_frequency(age, gender, migraine_type, eeg_abnormality):
    """
    Calculate optimal binaural beat frequency based on patient characteristics
    
    Args:
        age: Patient age
        gender: Patient gender (0=Male, 1=Female)
        migraine_type: 0=Control, 1=Aura, 2=Non-Aura
        eeg_abnormality: Dictionary from analyze_eeg_abnormality
        
    Returns:
        carrier_freq, beat_freq
    """
    # Base frequencies by migraine type
    if migraine_type == 1:  # Aura
        # Higher alpha band stimulation (calming hyperexcitability)
        base_carrier = 200
        base_beat = 10  # Alpha band
    elif migraine_type == 2:  # Non-Aura
        # Theta-alpha transition (relaxation)
        base_carrier = 150
        base_beat = 7  # Theta-alpha
    else:  # Control (for wellness)
        base_carrier = 180
        base_beat = 10  # Alpha
    
    # Adjust based on EEG abnormalities
    dominant_band = eeg_abnormality.get('dominant_band', 'normal')
    
    if dominant_band == 'delta':
        # High slow-wave activity → increase alpha stimulation
        base_beat = min(base_beat + 2, 12)
    elif dominant_band == 'theta':
        # Elevated theta → push toward alpha
        base_beat = min(base_beat + 1, 11)
    elif dominant_band == 'beta':
        # High beta → reduce toward theta-alpha
        base_beat = max(base_beat - 2, 6)
    
    # Age adjustments
    if age < 25:
        # Younger patients: slightly higher frequencies
        base_beat = min(base_beat + 0.5, 12)
    elif age > 40:
        # Older patients: slightly lower frequencies
        base_beat = max(base_beat - 0.5, 5)
    
    # Gender adjustments (subtle)
    if gender == 1:  # Female
        # Females may respond better to slightly higher alpha
        base_beat = min(base_beat + 0.3, 12)
    
    # Ensure beat frequency is in therapeutic range (4-12 Hz)
    beat_freq = np.clip(base_beat, 4, 12)
    
    return base_carrier, beat_freq


def generate_binaural_beat(carrier_freq, beat_freq, duration=600, sample_rate=44100):
    """
    Generate binaural beat audio
    
    Args:
        carrier_freq: Base carrier frequency (Hz)
        beat_freq: Beating frequency (Hz) - difference between ears
        duration: Duration in seconds (default 10 minutes)
        sample_rate: Audio sample rate (default 44.1 kHz)
        
    Returns:
        Stereo audio array (2, n_samples)
    """
    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Left ear: carrier frequency
    left_freq = carrier_freq
    left_channel = np.sin(2 * np.pi * left_freq * t)
    
    # Right ear: carrier + beat frequency
    right_freq = carrier_freq + beat_freq
    right_channel = np.sin(2 * np.pi * right_freq * t)
    
    # Apply fade in/out to avoid clicks
    fade_duration = 5  # seconds
    fade_samples = int(fade_duration * sample_rate)
    
    # Fade in
    fade_in = np.linspace(0, 1, fade_samples)
    left_channel[:fade_samples] *= fade_in
    right_channel[:fade_samples] *= fade_in
    
    # Fade out
    fade_out = np.linspace(1, 0, fade_samples)
    left_channel[-fade_samples:] *= fade_out
    right_channel[-fade_samples:] *= fade_out
    
    # Combine into stereo
    stereo = np.vstack([left_channel, right_channel])
    
    # Normalize to prevent clipping
    stereo = stereo / np.max(np.abs(stereo)) * 0.8
    
    return stereo


def save_audio(audio_data, patient_id, output_dir='output', sample_rate=44100):
    """
    Save audio to WAV file
    
    Args:
        audio_data: Audio array from generate_binaural_beat
        patient_id: Patient identifier
        output_dir: Output directory
        sample_rate: Sample rate
        
    Returns:
        filepath to saved audio
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    filename = f"{patient_id}_binaural_beat.wav"
    filepath = output_path / filename
    
    # Convert to int16 for WAV format
    audio_int16 = (audio_data.T * 32767).astype(np.int16)
    
    # Save to file
    wavfile.write(filepath, sample_rate, audio_int16)
    
    return filepath


def generate_treatment_report(patient_id, age, gender, migraine_type, 
                              carrier_freq, beat_freq, eeg_abnormality,
                              output_dir='output'):
    """
    Generate a text report explaining the treatment
    
    Args:
        patient_id: Patient identifier
        age, gender: Demographics
        migraine_type: Classification result
        carrier_freq, beat_freq: Generated frequencies
        eeg_abnormality: EEG analysis results
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    report_file = output_path / f"{patient_id}_treatment_report.txt"
    
    type_names = {0: 'Control', 1: 'Migraine with Aura', 2: 'Migraine without Aura'}
    gender_names = {0: 'Male', 1: 'Female'}
    
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PERSONALIZED BINAURAL BEAT THERAPY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Patient ID: {patient_id}\n")
        f.write(f"Age: {age}\n")
        f.write(f"Gender: {gender_names.get(gender, 'Unknown')}\n")
        f.write(f"Classification: {type_names.get(migraine_type, 'Unknown')}\n")
        f.write("\n" + "-" * 70 + "\n\n")
        
        f.write("EEG ANALYSIS:\n")
        f.write(f"  Dominant Band: {eeg_abnormality.get('dominant_band', 'N/A').upper()}\n")
        f.write(f"  Delta Ratio: {eeg_abnormality.get('delta_ratio', 0):.2%}\n")
        f.write(f"  Theta Ratio: {eeg_abnormality.get('theta_ratio', 0):.2%}\n")
        f.write(f"  Alpha Ratio: {eeg_abnormality.get('alpha_ratio', 0):.2%}\n")
        f.write(f"  Beta Ratio: {eeg_abnormality.get('beta_ratio', 0):.2%}\n")
        f.write("\n" + "-" * 70 + "\n\n")
        
        f.write("BINAURAL BEAT PRESCRIPTION:\n")
        f.write(f"  Carrier Frequency: {carrier_freq:.1f} Hz\n")
        f.write(f"  Beat Frequency: {beat_freq:.1f} Hz\n")
        f.write(f"  Target Brain State: ")
        
        if 4 <= beat_freq < 8:
            f.write("Theta (Deep Relaxation)\n")
        elif 8 <= beat_freq < 13:
            f.write("Alpha (Calm Focus)\n")
        else:
            f.write("Theta-Alpha Transition\n")
        
        f.write("\n" + "-" * 70 + "\n\n")
        
        f.write("USAGE INSTRUCTIONS:\n")
        f.write("  1. Use stereo headphones (binaural beats require separate ear channels)\n")
        f.write("  2. Listen for 10-15 minutes in a quiet, comfortable environment\n")
        f.write("  3. Recommended: Daily use, especially during prodrome phase\n")
        f.write("  4. Volume: Low to moderate (comfortable listening level)\n")
        f.write("  5. Do NOT listen while driving or operating machinery\n")
        f.write("\n" + "-" * 70 + "\n\n")
        
        f.write("DISCLAIMER:\n")
        f.write("  This is an experimental therapeutic approach based on EEG analysis\n")
        f.write("  and neuroscience research. It should complement, not replace,\n")
        f.write("  conventional medical treatment. Consult with a healthcare\n")
        f.write("  professional before use.\n")
        f.write("\n" + "=" * 70 + "\n")
    
    return report_file


if __name__ == "__main__":
    """Test binaural beat generation"""
    print("=" * 70)
    print("Testing Binaural Beat Generator")
    print("=" * 70)
    
    # Test parameters
    patient_id = "TEST_PATIENT"
    age = 25
    gender = 1  # Female
    migraine_type = 1  # Aura
    
    # Mock EEG abnormality
    eeg_abnormality = {
        'dominant_band': 'delta',
        'delta_ratio': 0.40,
        'theta_ratio': 0.25,
        'alpha_ratio': 0.20,
        'beta_ratio': 0.15
    }
    
    # Calculate optimal frequency
    print("\n1. Calculating optimal binaural beat frequency...")
    carrier_freq, beat_freq = calculate_optimal_frequency(
        age, gender, migraine_type, eeg_abnormality
    )
    print(f"   Carrier: {carrier_freq} Hz")
    print(f"   Beat: {beat_freq} Hz")
    
    # Generate audio
    print("\n2. Generating binaural beat (10 seconds for testing)...")
    audio = generate_binaural_beat(carrier_freq, beat_freq, duration=10)
    print(f"   Audio shape: {audio.shape}")
    print(f"   Duration: {audio.shape[1] / 44100:.1f} seconds")
    
    # Save audio
    print("\n3. Saving audio file...")
    audio_path = save_audio(audio, patient_id)
    print(f"   ✓ Saved to: {audio_path}")
    
    # Generate report
    print("\n4. Generating treatment report...")
    report_path = generate_treatment_report(
        patient_id, age, gender, migraine_type,
        carrier_freq, beat_freq, eeg_abnormality
    )
    print(f"   ✓ Report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("Binaural Beat Generator Test Complete!")
    print("=" * 70)
