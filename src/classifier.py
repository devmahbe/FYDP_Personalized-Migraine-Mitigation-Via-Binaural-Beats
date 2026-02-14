"""
Classifier Module
Train and evaluate migraine classification models
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def preprocess_features(X, y=None, scaler=None, pca=None, n_components=100):
    """
    Preprocess features: handle NaN, scale, and apply PCA
    
    Args:
        X: Feature matrix
        y: Labels (optional, for SMOTE)
        scaler: Fitted StandardScaler (if None, fit new one)
        pca: Fitted PCA (if None, fit new one)
        n_components: Number of PCA components
        
    Returns:
        X_processed, scaler, pca
    """
    # Handle NaN values (replace with median, or 0 if all NaN)
    X_clean = X.copy()
    for col in range(X_clean.shape[1]):
        col_data = X_clean[:, col]
        if np.isnan(col_data).any():
            median_val = np.nanmedian(col_data)
            # If entire column is NaN, median will be NaN, so use 0
            if np.isnan(median_val):
                median_val = 0.0
            X_clean[np.isnan(X_clean[:, col]), col] = median_val
    
    # Final check - replace any remaining NaNs with 0
    X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
    else:
        X_scaled = scaler.transform(X_clean)
    
    # Apply PCA for dimensionality reduction
    if pca is None:
        pca = PCA(n_components=min(n_components, X_scaled.shape[0], X_scaled.shape[1]))
        X_processed = pca.fit_transform(X_scaled)
    else:
        X_processed = pca.transform(X_scaled)
    
    return X_processed, scaler, pca


def train_classifier(X, y, model_type='random_forest', use_smote=True, n_components=50):
    """
    Train classifier with preprocessing pipeline
    
    Args:
        X: Feature matrix
        y: Labels
        model_type: 'random_forest' or 'svm'
        use_smote: Apply SMOTE for class balancing
        n_components: Number of PCA components
        
    Returns:
        Dictionary containing trained model and preprocessing objects
    """
    print("=" * 70)
    print(f"Training {model_type.upper()} Classifier")
    print("=" * 70)
    
    # Preprocess features
    print("\n1. Preprocessing features...")
    X_processed, scaler, pca = preprocess_features(X, y, n_components=n_components)
    print(f"   - Original features: {X.shape[1]}")
    print(f"   - After PCA: {X_processed.shape[1]}")
    print(f"   - Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Handle class imbalance with SMOTE
    if use_smote:
        print("\n2. Applying SMOTE for class balancing...")
        print(f"   Before SMOTE: {np.bincount(y)}")
        
        # Only apply SMOTE if we have enough samples
        min_samples = min(np.bincount(y))
        if min_samples >= 2:
            smote = SMOTE(random_state=42, k_neighbors=min(min_samples-1, 3))
            X_processed, y = smote.fit_resample(X_processed, y)
            print(f"   After SMOTE:  {np.bincount(y)}")
        else:
            print("   Skipping SMOTE (too few samples in minority class)")
    
    # Initialize classifier
    print(f"\n3. Training {model_type} model...")
    if model_type == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == 'svm':
        clf = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            random_state=42,
            class_weight='balanced',
            probability=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    clf.fit(X_processed, y)
    
    # Cross-validation
    print("\n4. Evaluating with cross-validation...")
    cv = StratifiedKFold(n_splits=min(3, min(np.bincount(y))), shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_processed, y, cv=cv, scoring='accuracy')
    print(f"   CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Package everything
    model_package = {
        'classifier': clf,
        'scaler': scaler,
        'pca': pca,
        'model_type': model_type,
        'n_components': n_components,
        'cv_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"\n✓ Model training complete!")
    
    return model_package


def evaluate_model(model_package, X_test, y_test):
    """
    Evaluate trained model on test set
    
    Args:
        model_package: Dictionary from train_classifier
        X_test, y_test: Test data
    """
    print("\n" + "=" * 70)
    print("Model Evaluation")
    print("=" * 70)
    
    # Preprocess test data
    X_processed, _, _ = preprocess_features(
        X_test,
        scaler=model_package['scaler'],
        pca=model_package['pca']
    )
    
    # Predict
    clf = model_package['classifier']
    y_pred = clf.predict(X_processed)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.3f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    target_names = ['Control', 'Aura', 'Non-Aura']
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    return accuracy, cm


def predict(model_package, X_new):
    """
    Predict class for new samples
    
    Args:
        model_package: Trained model package
        X_new: New feature matrix
        
    Returns:
        predictions, probabilities
    """
    # Preprocess
    X_processed, _, _ = preprocess_features(
        X_new,
        scaler=model_package['scaler'],
        pca=model_package['pca']
    )
    
    # Predict
    clf = model_package['classifier']
    predictions = clf.predict(X_processed)
    probabilities = clf.predict_proba(X_processed)
    
    return predictions, probabilities


def save_model(model_package, filepath='models/migraine_classifier.pkl'):
    """Save trained model"""
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True, parents=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"\n✓ Model saved to: {filepath}")


def load_model(filepath='models/migraine_classifier.pkl'):
    """Load trained model"""
    with open(filepath, 'rb') as f:
        model_package = pickle.load(f)
    
    return model_package


if __name__ == "__main__":
    """Train and evaluate classifier"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from dataset_builder import load_dataset
    
    print("=" * 70)
    print("Migraine Classification Model Training")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading dataset...")
    X, y, metadata = load_dataset(task='resting')
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    
    # Train model
    model_package = train_classifier(X_train, y_train, model_type='random_forest', n_components=30)
    
    # Evaluate
    evaluate_model(model_package, X_test, y_test)
    
    # Save model
    save_model(model_package)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
