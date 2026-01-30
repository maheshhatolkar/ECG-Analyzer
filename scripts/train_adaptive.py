"""Small utility to train/save adaptive Pan–Tompkins parameters.

This script is intentionally simple: it produces a reasonable default for
`integrated_multiplier` used by the detector. In practice you'd pass a set
of labeled ECG signals and pick the multiplier that best matches annotated
R-peak positions.

The produced JSON file can be loaded with `load_adaptive_params()` from
`ecg_analyzer.extractor` to influence detection thresholds.
"""

import argparse
import json
import os
import numpy as np

from ecg_analyzer.extractor import detect_r_peaks_pan_tompkin, refine_peaks_with_template


def compute_multiplier_from_signal(signal, fs):
    """Compute a baseline mean/std and return a starter multiplier.

    For now this uses a fixed heuristic: returns (mean, std, mult). A future
    implementation should optimize `mult` to match provided annotations.
    """
    from scipy.signal import butter, filtfilt
    low, high = 5.0, 15.0
    nyq = 0.5 * fs
    b, a = butter(1, [low / nyq, high / nyq], btype='band')
    filt = filtfilt(b, a, signal)
    diff = np.ediff1d(filt, to_end=0)
    squared = diff ** 2
    win = max(1, int(0.150 * fs))
    window = np.ones(win) / win
    integrated = np.convolve(squared, window, mode='same')
    mean = float(np.mean(integrated))
    std = float(np.std(integrated))
    # Simple default multiplier; replace with a search over labeled data later
    mult = 0.5
    return mean, std, mult


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', '-o', default='model_params.json')
    args = parser.parse_args()

    # Produce a modest default multiplier using a synthetic ECG signal
    fs = 250.0
    t = np.linspace(0, 10, int(fs * 10.0))
    sig = 0.02 * np.sin(2 * np.pi * 1.0 * t)
    for center in np.arange(0.5, 10, 1.0):
        idx = int(center * fs)
        sig += 0.8 * np.exp(-0.5 * ((np.arange(len(sig)) - idx) / (0.01 * fs)) ** 2)
    mean, std, mult = compute_multiplier_from_signal(sig, fs)
    params = {"integrated_multiplier": float(mult)}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
