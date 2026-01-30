import numpy as np
from ecg_analyzer.extractor import detect_r_peaks_pan_tompkin, detect_pqrst


def synthetic_signal(fs=250.0, duration=10.0):
    t = np.linspace(0, duration, int(fs * duration))
    sig = 0.02 * np.sin(2 * np.pi * 1.0 * t)
    # add R-peaks as Gaussians
    for center in np.arange(0.5, duration, 1.0):
        idx = int(center * fs)
        if 0 <= idx < len(sig):
            sig += 0.8 * np.exp(-0.5 * ((np.arange(len(sig)) - idx) / (0.01 * fs)) ** 2)
    return sig, fs


def test_r_peaks_and_pqrst():
    sig, fs = synthetic_signal()
    r = detect_r_peaks_pan_tompkin(sig, fs)
    assert len(r) >= 8
    pqrst = detect_pqrst(sig, r, fs)
    assert 'P' in pqrst and 'Q' in pqrst and 'T' in pqrst
    # check lengths match
    assert len(pqrst['R']) == len(pqrst['P'])
