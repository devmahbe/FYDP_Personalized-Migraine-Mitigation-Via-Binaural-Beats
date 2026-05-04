import json

def update_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_code = """def detect_bad_channels(raw,
                        flatline_threshold_s=5,
                        zscore_threshold=4,
                        min_correlation=0.75,
                        correlation_window_s=4.0,
                        good_window_min_frac=0.80):
    \"\"\"
    Detect bad EEG channels — three criteria from DISCOVER-EEG paper.
    \"\"\"
    import numpy as np
    import mne
    
    bad_channels = []
    reasons = {}

    # BUG FIX 1: Provide exclude=[] to consider all channels every run
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, exclude=[])
    all_eeg_names = [raw.ch_names[p] for p in picks_eeg]

    # exclude FCz it is the original reference electrode (all-zeros signal).
    eeg_names = [ch for ch in all_eeg_names if ch != 'FCz']
    data_picks = [picks_eeg[all_eeg_names.index(ch)] for ch in eeg_names]
    data = raw.get_data(picks=data_picks)     
    sfreq = raw.info['sfreq']

    # ── Criterion 1: Flatline (dead electrode) ───────────────────────────
    flatline_samples = int(flatline_threshold_s * sfreq)
    step = max(1, flatline_samples // 2)
    for i, ch_name in enumerate(eeg_names):
        for start in range(0, len(data[i]) - flatline_samples, step):
            if np.ptp(data[i, start:start + flatline_samples]) < 1e-8:
                bad_channels.append(ch_name)
                reasons[ch_name] = 'flatline (dead electrode)'
                break

    # ── Criterion 2: Extreme variance (z-score) ──────────────────────────
    channel_vars = np.var(data, axis=1)
    if np.std(channel_vars) > 0:
        z_scores = (channel_vars - np.mean(channel_vars)) / np.std(channel_vars)
        for i, ch_name in enumerate(eeg_names):
            if z_scores[i] > zscore_threshold and ch_name not in bad_channels:
                bad_channels.append(ch_name)
                reasons[ch_name] = f'extreme variance (z={z_scores[i]:.2f}, threshold={zscore_threshold})'

    # ── Criterion 3: Low spatial correlation ─────────────────────────────
    # BUG FIX 2: Check correlation with closest neighbours instead of a global random average
    reference_set = [j for j, ch in enumerate(eeg_names) if ch not in bad_channels]

    if len(reference_set) >= 5:
        window_samples = int(correlation_window_s * sfreq)
        n_windows = max(1, data.shape[1] // window_samples)

        for i, ch_name in enumerate(eeg_names):
            if ch_name in bad_channels:
                continue
            
            good_windows = 0
            other_idx = [j for j in reference_set if j != i]
            
            if len(other_idx) < 5:
                continue
                
            for w in range(n_windows):
                s = w * window_samples
                e = s + window_samples
                
                # Compute correlation with all other good channels in this window
                subset_data = data[other_idx, s:e]
                target_data = data[i, s:e]
                
                # Fast row-wise correlation manually to avoid huge corrcoef matrix
                target_zm = target_data - np.mean(target_data)
                subset_zm = subset_data - np.mean(subset_data, axis=1, keepdims=True)
                
                target_std = np.sqrt(np.sum(target_zm**2))
                subset_std = np.sqrt(np.sum(subset_zm**2, axis=1))
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    corrs = np.abs(np.sum(subset_zm * target_zm, axis=1) / (target_std * subset_std))
                
                corrs = np.nan_to_num(corrs, nan=0.0)
                
                # 95th percentile correlation isolates its closest functional neighbours
                local_corr = np.percentile(corrs, 95)
                
                if local_corr >= min_correlation:
                    good_windows += 1
                    
            pct_good = good_windows / n_windows
            # Channel must be valid in AT LEAST good_window_min_frac of the recording
            if pct_good < good_window_min_frac:
                bad_channels.append(ch_name)
                reasons[ch_name] = f'low spatial correlation ({pct_good*100:.0f}% of windows >= {min_correlation})'

    return bad_channels, reasons

print('Detecting bad channels...')
print(f'  Criteria & thresholds:')
print(f'    1. Flatline       > 5 s continuous')
print(f'    2. Z-score noise  > 4 SD above all-channel mean')
print(f'    3. Correlation    < 0.75 locally in >20% of 4-s windows')
print(f'    (FCz excluded - it is the recording reference electrode)')

bad_chs, bad_reasons = detect_bad_channels(raw)

print(f'\\nResult: {len(bad_chs)} bad channel(s) detected')
for ch in bad_chs:
    print(f'  - {ch}: {bad_reasons[ch]}')

raw.info['bads'] = bad_chs
n_eeg_total = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
print(f'\\n(Paper reports: ~1-2 bad channels per recording, ~2-3%)')
print(f'We found {len(bad_chs)}/{n_eeg_total} ({len(bad_chs)/n_eeg_total*100:.1f}%)')
"""

    # Convert to array of strings per jupyter format
    lines = new_code.split('\\n')
    source = [line + '\\n' for line in lines[:-1]] + [lines[-1]]
    
    modified = False
    for cell in nb['cells']:
        src = ''.join(cell.get('source', []))
        if 'def detect_bad_channels' in src or 'bad_chs, bad_reasons = detect_bad_channels' in src:
            cell['source'] = source
            modified = True
            break
            
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print('Successfully modified the notebook.')
    else:
        print('Cell not found!')

if __name__ == "__main__":
    update_notebook(r'G:\Study\FYDP-I_Personalized-Migraine-Mitigation-Via-Binaural-Beats\LEMON_Preprocessing_Pipeline.ipynb')
