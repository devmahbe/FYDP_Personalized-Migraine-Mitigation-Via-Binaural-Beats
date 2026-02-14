"""
Main Pipeline Module
End-to-end pipeline for migraine classification and binaural beat generation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from pathlib import Path

from data_loader import load_clinical_data, get_patient_label
from feature_extraction import extract_all_features
from classifier import load_model, predict
from binaural_beat_generator import (
    analyze_eeg_abnormality,
    calculate_optimal_frequency,
    generate_binaural_beat,
    save_audio,
    generate_treatment_report
)


def predict_and_treat(patient_id, model_path='models/migraine_classifier.pkl', 
                     duration=600, output_dir='output', verbose=True):
    """
    Complete pipeline: classify patient and generate personalized binaural beat
    
    Args:
        patient_id: Patient identifier (e.g., 'M1_1', 'C1')
        model_path: Path to trained classifier
        duration: Audio duration in seconds (default 10 minutes)
        output_dir: Output directory
        verbose: Print progress
        
    Returns:
        Dictionary with results
    """
    if verbose:
        print("=" * 70)
        print(f"Processing Patient: {patient_id}")
        print("=" * 70)
    
    # 1. Load clinical data
    if verbose:
        print("\n[1/6] Loading clinical data...")
    
    clinical_df = load_clinical_data()
    base_id = patient_id.split('_')[0]
    patient_row = clinical_df[clinical_df['P#'] == base_id]
    
    if len(patient_row) == 0:
        raise ValueError(f"Patient {base_id} not found in clinical database")
    
    age = patient_row['Age'].values[0]
    gender_encoded = patient_row['gender_encoded'].values[0]
    true_label = patient_row['label'].values[0]
    
    if verbose:
        print(f"   Age: {age}, Gender: {['Male', 'Female'][gender_encoded]}")
        print(f"   True label: {['Control', 'Aura', 'Non-Aura'][true_label]}")
    
    # 2. Extract EEG features
    if verbose:
        print("\n[2/6] Extracting EEG features...")
    
    features = extract_all_features(patient_id, task='resting', verbose=False)
    
    if features is None:
        raise ValueError(f"Could not extract features for {patient_id}")
    
    # Add clinical features
    clinical_features = np.array([age, gender_encoded])
    combined_features = np.concatenate([features, clinical_features])
    
    if verbose:
        print(f"   ✓ Extracted {len(combined_features)} features")
    
    # 3. Load classifier and predict
    if verbose:
        print("\n[3/6] Classifying migraine type...")
    
    model = load_model(model_path)
    X_new = combined_features.reshape(1, -1)
    predictions, probabilities = predict(model, X_new)
    
    predicted_label = predictions[0]
    pred_proba = probabilities[0]
    
    label_names = ['Control', 'Aura', 'Non-Aura']
    
    if verbose:
        print(f"   Predicted: {label_names[predicted_label]}")
        print(f"   Confidence: Control={pred_proba[0]:.1%}, Aura={pred_proba[1]:.1%}, Non-Aura={pred_proba[2]:.1%}")
    
    # 4. Analyze EEG abnormality
    if verbose:
        print("\n[4/6] Analyzing EEG abnormalities...")
    
    eeg_abnormality = analyze_eeg_abnormality(features)
    
    if verbose:
        print(f"   Dominant band: {eeg_abnormality['dominant_band']}")
    
    # 5. Calculate optimal binaural beat frequency
    if verbose:
        print("\n[5/6] Calculating therapeutic frequency...")
    
    carrier_freq, beat_freq = calculate_optimal_frequency(
        age, gender_encoded, predicted_label, eeg_abnormality
    )
    
    if verbose:
        print(f"   Carrier: {carrier_freq} Hz")
        print(f"   Beat: {beat_freq:.1f} Hz")
    
    # 6. Generate binaural beat and save
    if verbose:
        print(f"\n[6/6] Generating binaural beat ({duration}s)...")
    
    audio = generate_binaural_beat(carrier_freq, beat_freq, duration=duration)
    audio_path = save_audio(audio, patient_id, output_dir=output_dir)
    
    if verbose:
        print(f"   ✓ Audio saved: {audio_path}")
    
    # Generate treatment report
    report_path = generate_treatment_report(
        patient_id, age, gender_encoded, predicted_label,
        carrier_freq, beat_freq, eeg_abnormality,
        output_dir=output_dir
    )
    
    if verbose:
        print(f"   ✓ Report saved: {report_path}")
    
    # Return results
    results = {
        'patient_id': patient_id,
        'age': age,
        'gender': gender_encoded,
        'true_label': true_label,
        'predicted_label': predicted_label,
        'probabilities': pred_proba,
        'carrier_freq': carrier_freq,
        'beat_freq': beat_freq,
        'eeg_abnormality': eeg_abnormality,
        'audio_path': str(audio_path),
        'report_path': str(report_path)
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("✓ Processing Complete!")
        print("=" * 70)
    
    return results


def batch_process(patient_ids, **kwargs):
    """
    Process multiple patients
    
    Args:
        patient_ids: List of patient IDs
        **kwargs: Arguments for predict_and_treat
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    for idx, patient_id in enumerate(patient_ids):
        print(f"\n{'='*70}")
        print(f"Patient {idx+1}/{len(patient_ids)}: {patient_id}")
        print(f"{'='*70}")
        
        try:
            result = predict_and_treat(patient_id, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"✗ Error processing {patient_id}: {e}")
            results.append({'patient_id': patient_id, 'error': str(e)})
    
    return results


if __name__ == "__main__":
    """Test the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', type=str, default='M1_1',
                       help='Patient ID to process')
    parser.add_argument('--duration', type=int, default=60,
                       help='Audio duration in seconds')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple patients from dataset')
    args = parser.parse_args()
    
    if args.batch:
        # Process first few patients from dataset
        from dataset_builder import load_dataset
        _, _, metadata = load_dataset(task='resting')
        patient_ids = metadata['patient_id'].tolist()[:5]
        
        print(f"Processing {len(patient_ids)} patients...")
        results = batch_process(patient_ids, duration=args.duration, verbose=True)
        
        print("\n" + "=" * 70)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 70)
        for r in results:
            if 'error' not in r:
                label_names = ['Control', 'Aura', 'Non-Aura']
                print(f"{r['patient_id']}: Predicted={label_names[r['predicted_label']]}, "
                      f"Beat={r['beat_freq']:.1f}Hz")
    else:
        # Process single patient
        result = predict_and_treat(args.patient, duration=args.duration, verbose=True)
