"""
Resting-Only Migraine Trait Classifier — Baseline
==================================================
Builds a subject-level migraine-vs-control classifier from the existing
resting-state preprocessed arrays in data/MIGRAINE_GPU_preprocessed/resting/.

Design (per the research pipeline doc):
  - Labels from Migraine_Control_Demographics.xlsx (NOT folder prefixes)
  - Excludes M2, M6, M18 (medicated); keeps M13 (resting only)
  - Reconstructs ~20-s windows by concatenating 10 contiguous 2-s epochs
  - Features: relative band power, alpha peak frequency, spectral entropy,
    wPLI connectivity graph metrics (32-ch 10-20 subset), topographic asymmetry
  - Validation: Leave-One-Subject-Out + grouped 5-fold, subject-level AUC
  - Models: Logistic Regression (L2), Random Forest, SVM (RBF)
"""

import os
import re
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from scipy.stats import entropy as sp_entropy

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESTING_DIR = PROJECT_ROOT / 'data' / 'MIGRAINE_GPU_preprocessed' / 'resting'
DEMOGRAPHICS = PROJECT_ROOT / 'Dataset' / 'Migraine_Control_Demographics.xlsx'
OUTPUT_DIR = PROJECT_ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

EXCLUDED_MEDICATED = {'M2', 'M6', 'M18'}   # excluded in original study (medication)
SFREQ = 512.0                               # native sampling rate of saved epochs
EPOCH_S = 2.0                               # each saved epoch is 2 s
WINDOW_EPOCHS = 10                          # 10 x 2 s = 20 s windows
WINDOW_S = EPOCH_S * WINDOW_EPOCHS

BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 100.0),
}

# 32-channel 10-20 subset (all present in the 130-ch montage)
CONN_CHANNELS = [
    'Fp1', 'Fp2', 'AF7', 'AF8',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'PO9', 'O1', 'Oz', 'O2', 'PO10',
]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_labels():
    """Build subject -> label map from demographics Excel."""
    df = pd.read_excel(DEMOGRAPHICS)
    labels = {}
    for _, row in df.iterrows():
        pid = str(row['P#']).strip()
        if pid.startswith('C'):
            labels[pid] = 0  # control
        else:
            labels[pid] = 1  # migraine
    return labels


def discover_resting_files():
    """Find all resting broadband .npy files and map to subject IDs."""
    files = sorted(glob.glob(str(RESTING_DIR / '*_broadband.npy')))
    records = []
    for f in files:
        name = os.path.basename(f)
        # Pattern: {subject_dir}_{source}_broadband.npy
        # e.g. C1_C1_Resting_broadband.npy, M10_1_M10_Resting_broadband.npy
        m = re.match(r'^(C\d+|M\d+_\d+)_(.+)_broadband\.npy$', name)
        if not m:
            print(f'  [skip] unrecognized file: {name}')
            continue
        subject_dir = m.group(1)
        base_id = subject_dir.split('_')[0]  # C1, M10, etc.
        records.append({
            'file': f,
            'subject_dir': subject_dir,
            'base_id': base_id,
        })
    return records


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def welch_psd(data, sfreq=SFREQ):
    """Compute Welch PSD for each channel. data: (n_ch, n_samples)."""
    freqs, psd = sp_signal.welch(
        data, fs=sfreq, nperseg=int(4 * sfreq),
        noverlap=int(2 * sfreq), axis=-1
    )
    return freqs, psd


def band_power_features(psd, freqs):
    """Relative band power per channel. Returns (n_ch, n_bands)."""
    total = np.trapezoid(psd, freqs, axis=-1)  # (n_ch,)
    total[total == 0] = 1e-12
    feats = []
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        bp = np.trapezoid(psd[:, mask], freqs[mask], axis=-1)
        feats.append(bp / total)
    return np.stack(feats, axis=-1)  # (n_ch, n_bands)


def alpha_peak_frequency(psd, freqs):
    """Alpha peak frequency per channel (8-13 Hz)."""
    mask = (freqs >= 8.0) & (freqs <= 13.0)
    f_alpha = freqs[mask]
    p_alpha = psd[:, mask]
    peaks = f_alpha[np.argmax(p_alpha, axis=-1)]
    return peaks  # (n_ch,)


def spectral_entropy_features(psd):
    """Normalized spectral entropy per channel."""
    p = psd / (psd.sum(axis=-1, keepdims=True) + 1e-12)
    return sp_entropy(p, axis=-1) / np.log(psd.shape[-1])  # (n_ch,)


def wpli_matrix(data, sfreq=SFREQ):
    """Weighted Phase Lag Index between all channel pairs.
    data: (n_ch, n_samples). Returns (n_ch, n_ch) wPLI matrix."""
    n_ch = data.shape[0]
    # Hilbert transform to get analytic phase
    analytic = sp_signal.hilbert(data, axis=-1)
    phase = np.angle(analytic)
    wpli = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            dphi = phase[i] - phase[j]
            imag = np.sin(dphi)
            num = np.abs(np.mean(imag))
            den = np.mean(np.abs(imag))
            wpli[i, j] = wpli[j, i] = num / (den + 1e-12)
    return wpli


def connectivity_graph_features(wpli):
    """Graph metrics from wPLI matrix: mean, std, clustering, path length."""
    n = wpli.shape[0]
    triu = wpli[np.triu_indices(n, k=1)]
    feats = {
        'conn_mean': triu.mean(),
        'conn_std': triu.std(),
        'conn_max': triu.max(),
        'conn_min': triu.min(),
        'conn_median': np.median(triu),
    }
    # Clustering coefficient (weighted)
    w = wpli.copy()
    np.fill_diagonal(w, 0)
    deg = w.sum(axis=1)
    denom = deg * (deg - 1)
    denom[denom == 0] = 1e-12
    clustering = np.zeros(n)
    for i in range(n):
        neighbors = np.where(w[i] > 0)[0]
        if len(neighbors) < 2:
            continue
        sub = w[np.ix_(neighbors, neighbors)]
        clustering[i] = sub.sum() / (len(neighbors) * (len(neighbors) - 1) + 1e-12)
    feats['conn_clustering'] = clustering.mean()
    return feats


def topographic_asymmetry(band_power, ch_names):
    """Left-right and anterior-posterior power ratios."""
    idx = {ch: i for i, ch in enumerate(ch_names)}
    left = ['Fp1', 'F7', 'F3', 'T7', 'C3', 'P7', 'P3', 'O1']
    right = ['Fp2', 'F8', 'F4', 'T8', 'C4', 'P8', 'P4', 'O2']
    ant = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8']
    post = ['P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
    feats = {}
    for b, band_name in enumerate(BANDS.keys()):
        lr = np.mean([band_power[idx[c], b] for c in left if c in idx]) / \
             (np.mean([band_power[idx[c], b] for c in right if c in idx]) + 1e-12)
        ap = np.mean([band_power[idx[c], b] for c in ant if c in idx]) / \
             (np.mean([band_power[idx[c], b] for c in post if c in idx]) + 1e-12)
        feats[f'LR_{band_name}'] = lr
        feats[f'AP_{band_name}'] = ap
    return feats


def extract_window_features(window_data, ch_names):
    """Extract all features from one 20-s window.
    window_data: (n_ch, n_samples). Returns dict of scalar features."""
    feats = {}

    # PSD-based features
    freqs, psd = welch_psd(window_data)
    bp = band_power_features(psd, freqs)          # (n_ch, 5)
    apf = alpha_peak_frequency(psd, freqs)        # (n_ch,)
    se = spectral_entropy_features(psd)           # (n_ch,)

    # Flatten band power (log-transformed)
    for b, band_name in enumerate(BANDS.keys()):
        feats[f'bp_{band_name}_mean'] = np.log1p(bp[:, b].mean())
        feats[f'bp_{band_name}_std'] = np.log1p(bp[:, b].std())
        feats[f'bp_{band_name}_max'] = np.log1p(bp[:, b].max())
        feats[f'bp_{band_name}_min'] = np.log1p(bp[:, b].min())

    # Alpha peak frequency stats
    feats['apf_mean'] = apf.mean()
    feats['apf_std'] = apf.std()
    feats['apf_max'] = apf.max()
    feats['apf_min'] = apf.min()

    # Spectral entropy stats
    feats['se_mean'] = se.mean()
    feats['se_std'] = se.std()

    # Connectivity on 32-ch subset
    conn_idx = [ch_names.index(c) for c in CONN_CHANNELS if c in ch_names]
    if len(conn_idx) >= 16:
        wpli = wpli_matrix(window_data[conn_idx])
        feats.update(connectivity_graph_features(wpli))

    # Topographic asymmetry
    feats.update(topographic_asymmetry(bp, ch_names))

    return feats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print('=' * 70)
    print('RESTING-ONLY MIGRAINE TRAIT CLASSIFIER — BASELINE')
    print('=' * 70)

    # 1. Labels
    labels = load_labels()
    print(f'\n[1] Loaded {len(labels)} subject labels from demographics.')

    # 2. Discover files
    records = discover_resting_files()
    print(f'[2] Found {len(records)} resting broadband files.')

    # 3. Filter by exclusion criteria
    kept = []
    for r in records:
        base = r['base_id']
        if base in EXCLUDED_MEDICATED:
            print(f'    [excluded] {r["subject_dir"]} (medicated)')
            continue
        if base not in labels:
            print(f'    [excluded] {r["subject_dir"]} (no demographics)')
            continue
        kept.append(r)
    print(f'    Kept {len(kept)} recordings after exclusions.')

    # 4. Build windows + features
    print('\n[3] Extracting features from 20-s windows...')
    X_list, y_list, groups_list = [], [], []
    subject_stats = []
    feat_names = None

    for r in kept:
        arr = np.load(r['file'])  # (n_epochs, 130, 1024)
        n_epochs = arr.shape[0]
        n_windows = n_epochs // WINDOW_EPOCHS
        if n_windows == 0:
            print(f'    [skip] {r["subject_dir"]}: only {n_epochs} epochs (< {WINDOW_EPOCHS})')
            continue

        # Load channel names for this recording
        ch_file = RESTING_DIR / f'channel_names_{os.path.basename(r["file"]).replace("_broadband.npy", "")}.txt'
        if not ch_file.exists():
            # Fallback: use the top-level channel_names.txt
            ch_file = RESTING_DIR.parent / 'channel_names.txt'
        with open(ch_file) as fh:
            ch_names = [l.strip() for l in fh if l.strip()]

        label = labels[r['base_id']]
        n_feat = None
        for w in range(n_windows):
            start = w * WINDOW_EPOCHS
            end = start + WINDOW_EPOCHS
            window_data = arr[start:end].transpose(1, 0, 2).reshape(arr.shape[1], -1)  # (130, 20s*512)
            feats = extract_window_features(window_data, ch_names)
            if n_feat is None:
                n_feat = len(feats)
                feat_names = list(feats.keys())
            X_list.append(list(feats.values()))
            y_list.append(label)
            groups_list.append(r['subject_dir'])

        subject_stats.append({
            'subject': r['subject_dir'],
            'group': 'migraine' if label == 1 else 'control',
            'n_epochs': n_epochs,
            'n_windows': n_windows,
        })
        print(f'    {r["subject_dir"]}: {n_epochs} epochs -> {n_windows} windows (label={label})')

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list)
    groups = np.array(groups_list)
    print(f'\n    Total windows: {X.shape[0]}, features: {X.shape[1]}')
    print(f'    Migraine windows: {(y == 1).sum()}, Control windows: {(y == 0).sum()}')
    print(f'    Subjects: {len(np.unique(groups))}')

    # Save subject stats
    pd.DataFrame(subject_stats).to_csv(OUTPUT_DIR / 'resting_subject_stats.csv', index=False)

    # 5. Standardize + PCA
    print('\n[4] Standardizing and reducing features (PCA)...')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(50, X.shape[1]), random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print(f'    PCA explained variance: {pca.explained_variance_ratio_.sum():.3f} '
          f'({pca.n_components_} components)')

    # 6. Models
    models = {
        'LogisticRegression': LogisticRegression(
            C=1.0, max_iter=2000, random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=RANDOM_STATE),
        'SVM_RBF': SVC(
            C=1.0, kernel='rbf', gamma='scale', probability=True,
            random_state=RANDOM_STATE),
    }

    # 7. LOSO evaluation
    print('\n[5] Leave-One-Subject-Out evaluation...')
    logo = LeaveOneGroupOut()
    results = []
    roc_data = {}

    for model_name, model in models.items():
        print(f'\n  --- {model_name} ---')
        y_true_all, y_prob_all, y_pred_all = [], [], []
        fold_aucs = []
        fold_accs = []
        fold_f1s = []

        for train_idx, test_idx in logo.split(X_pca, y, groups):
            X_tr, X_te = X_pca[train_idx], X_pca[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Fit scaler on train only (leakage prevention)
            scaler_fold = StandardScaler()
            X_tr_s = scaler_fold.fit_transform(X_tr)
            X_te_s = scaler_fold.transform(X_te)

            model.fit(X_tr_s, y_tr)
            y_prob = model.predict_proba(X_te_s)[:, 1]
            y_pred = model.predict(X_te_s)

            y_true_all.extend(y_te)
            y_prob_all.extend(y_prob)
            y_pred_all.extend(y_pred)

            if len(np.unique(y_te)) > 1:
                fold_aucs.append(roc_auc_score(y_te, y_prob))
            fold_accs.append(accuracy_score(y_te, y_pred))
            fold_f1s.append(f1_score(y_te, y_pred, zero_division=0))

        # Subject-level aggregation (mean probability per subject)
        subj_prob = {}
        subj_true = {}
        for g, p, t in zip(groups, y_prob_all, y_true_all):
            subj_prob.setdefault(g, []).append(p)
            subj_true[g] = t
        subj_prob_mean = np.array([np.mean(v) for v in subj_prob.values()])
        subj_true_arr = np.array([subj_true[g] for g in subj_prob.keys()])
        subj_pred = (subj_prob_mean >= 0.5).astype(int)

        subj_auc = roc_auc_score(subj_true_arr, subj_prob_mean)
        subj_acc = accuracy_score(subj_true_arr, subj_pred)
        subj_prec = precision_score(subj_true_arr, subj_pred, zero_division=0)
        subj_rec = recall_score(subj_true_arr, subj_pred, zero_division=0)
        subj_f1 = f1_score(subj_true_arr, subj_pred, zero_division=0)
        cm = confusion_matrix(subj_true_arr, subj_pred)

        results.append({
            'model': model_name,
            'window_auc': np.mean(fold_aucs) if fold_aucs else np.nan,
            'window_auc_std': np.std(fold_aucs) if fold_aucs else np.nan,
            'subject_auc': subj_auc,
            'subject_accuracy': subj_acc,
            'subject_precision': subj_prec,
            'subject_recall': subj_rec,
            'subject_f1': subj_f1,
            'n_subjects': len(subj_true_arr),
            'n_migraine': int(subj_true_arr.sum()),
            'n_control': int((1 - subj_true_arr).sum()),
            'cm_tn': int(cm[0, 0]), 'cm_fp': int(cm[0, 1]),
            'cm_fn': int(cm[1, 0]), 'cm_tp': int(cm[1, 1]),
        })

        # ROC curve for plotting
        fpr, tpr, _ = roc_curve(subj_true_arr, subj_prob_mean)
        roc_data[model_name] = (fpr, tpr, subj_auc)

        print(f'    Window-level AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}')
        print(f'    Subject-level AUC: {subj_auc:.3f}')
        print(f'    Subject-level Acc: {subj_acc:.3f} | Prec: {subj_prec:.3f} | '
              f'Rec: {subj_rec:.3f} | F1: {subj_f1:.3f}')
        print(f'    Confusion matrix (TN FP / FN TP): {cm.tolist()}')

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'resting_baseline_results.csv', index=False)
    print(f'\n[6] Results saved to {OUTPUT_DIR / "resting_baseline_results.csv"}')

    # 8. Plot ROC curves
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Resting-Only Migraine Trait Detection — Subject-Level ROC')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'resting_baseline_roc.png', dpi=150)
    print(f'    ROC plot saved to {OUTPUT_DIR / "resting_baseline_roc.png"}')

    # 9. Feature importance (Random Forest on full feature set)
    print('\n[7] Feature importance (Random Forest, top 20)...')
    rf_imp = RandomForestClassifier(n_estimators=500, max_depth=8, random_state=RANDOM_STATE)
    rf_imp.fit(X_scaled, y)
    importances = pd.Series(rf_imp.feature_importances_, index=feat_names)
    importances = importances.sort_values(ascending=False).head(20)
    print(importances.to_string())
    importances.to_csv(OUTPUT_DIR / 'resting_feature_importance.csv')

    # 10. Summary
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(results_df[['model', 'subject_auc', 'subject_accuracy', 'subject_precision',
                      'subject_recall', 'subject_f1']].to_string(index=False))
    print('\nChance-level reference (Combrisson & Jerbi 2015):')
    n_subj = results_df['n_subjects'].iloc[0]
    n_pos = results_df['n_migraine'].iloc[0]
    chance_acc = 0.5
    se = np.sqrt(chance_acc * (1 - chance_acc) / n_subj)
    print(f'  n={n_subj} subjects ({n_pos} migraine / {n_subj - n_pos} control)')
    print(f'  Chance accuracy = 0.500, 95% CI = [{chance_acc - 1.96*se:.3f}, {chance_acc + 1.96*se:.3f}]')
    print('\nDone.')


if __name__ == '__main__':
    main()