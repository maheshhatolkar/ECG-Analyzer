import os
import json
import csv
import tempfile
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import cv2
from PIL import Image
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
import math

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

# Adaptive parameters (can be trained/saved)
ADAPTIVE_PARAMS = {"integrated_multiplier": 0.5}


def save_adaptive_params(path: str):
    import json
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(ADAPTIVE_PARAMS, f, indent=2)


def load_adaptive_params(path: str):
    import json, os
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    ADAPTIVE_PARAMS.update(params)


def load_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if convert_from_path is None:
            raise RuntimeError("pdf2image not available; install pdf2image and poppler")
        pages = convert_from_path(path, dpi=200)
        pil = pages[0]
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        pil = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def validate_image_quality(img: np.ndarray) -> Tuple[bool, str]:
    h, w = img.shape[:2]
    if w * h < 20000:
        return False, "Image resolution too small"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nonzero_ratio = np.count_nonzero(gray < 250) / (w * h)
    if nonzero_ratio < 0.001:
        return False, "Image appears blank or overexposed"
    return True, "OK"


def deskew_image(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    angle = 0.0
    if lines is not None:
        angles = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            a = np.arctan2(y2 - y1, x2 - x1)
            angles.append(a)
        if angles:
            median_angle = np.median(angles)
            angle = median_angle * 180.0 / np.pi
    if abs(angle) < 0.1:
        return img
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def enhance_image(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced


def remove_gridlines(gray: np.ndarray) -> np.ndarray:
    # Morphological approach to reduce gridlines while preserving waveform
    img_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)
    # remove thin lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    removed_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_h)
    removed_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_v)
    grid = cv2.bitwise_or(removed_h, removed_v)
    cleaned = cv2.bitwise_and(img_bin, cv2.bitwise_not(grid))
    return cleaned


def extract_waveform(cleaned: np.ndarray) -> Tuple[np.ndarray, float]:
    h, w = cleaned.shape
    xs = []
    ys = []
    for x in range(w):
        col = cleaned[:, x]
        inds = np.where(col > 0)[0]
        if inds.size == 0:
            xs.append(x)
            ys.append(np.nan)
        else:
            y = np.median(inds)
            xs.append(x)
            ys.append(y)
    ys = np.array(ys)
    # interpolate missing
    nans = np.isnan(ys)
    if nans.any():
        ys[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), ys[~nans])
    signal = h - ys  # invert to make upward positive
    # normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)
    # try to detect grid spacing to calculate seconds per pixel
    seconds_per_pixel = detect_grid_seconds_per_pixel(cleaned)
    if seconds_per_pixel is None:
        # fallback conservative guess (250 Hz implies 0.004s)
        seconds_per_pixel = 0.004
    return signal, seconds_per_pixel


def detect_grid_seconds_per_pixel(cleaned: np.ndarray) -> Optional[float]:
    # Attempt to find regular vertical grid spacing (pixels per small box)
    proj = np.mean(cleaned, axis=0)
    # peaks correspond to gridlines (dark lines)
    peaks, _ = find_peaks(proj, distance=3)
    if peaks.size < 2:
        return None
    diffs = np.diff(peaks)
    if diffs.size == 0:
        return None
    median_spacing = int(np.median(diffs))
    if median_spacing <= 0:
        return None
    # ECG small box ~1 mm -> 0.04s at 25 mm/s paper speed
    seconds_per_pixel = 0.04 / float(median_spacing)
    return seconds_per_pixel


def bandpass(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(2, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def detect_r_peaks_pan_tompkin(signal: np.ndarray, fs: float) -> np.ndarray:
    # Pan-Tompkins style detector (simplified)
    # 1) Bandpass around QRS
    low, high = 5.0, 15.0
    nyq = 0.5 * fs
    b, a = butter(1, [low / nyq, high / nyq], btype='band')
    filt = filtfilt(b, a, signal)
    # 2) derivative
    diff = np.ediff1d(filt, to_end=0)
    # 3) squaring
    squared = diff ** 2
    # 4) moving window integration
    win = int(0.150 * fs) if fs * 0.150 >= 1 else 1
    window = np.ones(win) / win
    integrated = np.convolve(squared, window, mode='same')
    # 5) detect peaks on integrated signal
    distance = max(1, int(0.25 * fs))
    # adaptive thresholding using global multiplier
    mult = ADAPTIVE_PARAMS.get('integrated_multiplier', 0.5)
    threshold = float(np.mean(integrated) + mult * np.std(integrated))
    peaks_i, _ = find_peaks(integrated, distance=distance, height=threshold)
    # refine peaks on original filtered signal: pick local max within +/- 0.05s
    refined = []
    search_radius = int(0.05 * fs)
    for p in peaks_i:
        lo = max(0, p - search_radius)
        hi = min(len(filt) - 1, p + search_radius)
        local_peak = lo + np.argmax(np.abs(filt[lo:hi + 1]))
        refined.append(local_peak)
    if len(refined) == 0:
        return np.array([], dtype=int)
    peaks = np.array(sorted(set(refined)), dtype=int)
    return peaks


def refine_peaks_with_template(signal: np.ndarray, r_peaks: np.ndarray, fs: float, half_window_ms: int = 60) -> np.ndarray:
    """Refine R-peak indices using cross-correlation with an averaged template."""
    if len(r_peaks) == 0:
        return r_peaks
    hw = max(1, int((half_window_ms / 1000.0) * fs))
    beats = []
    for r in r_peaks:
        lo = max(0, r - hw)
        hi = min(len(signal), r + hw)
        seg = signal[lo:hi]
        if len(seg) == 2 * hw:
            beats.append(seg)
    if len(beats) < 3:
        return r_peaks
    # align by max within each beat
    aligned = []
    for seg in beats:
        shift = np.argmax(np.abs(seg))
        # center the peak
        pad_left = hw - shift
        if pad_left >= 0:
            aligned_seg = np.pad(seg, (pad_left, 0), mode='constant')[:2 * hw]
        else:
            aligned_seg = seg[-pad_left:]
            aligned_seg = np.pad(aligned_seg, (0, 2 * hw - len(aligned_seg)), mode='constant')
        aligned.append(aligned_seg)
    template = np.mean(np.vstack(aligned), axis=0)

    refined = []
    for r in r_peaks:
        lo = max(0, r - hw)
        hi = min(len(signal), r + hw)
        seg = signal[lo:hi]
        if seg.size < template.size:
            # pad
            seg = np.pad(seg, (0, template.size - seg.size), mode='constant')
        corr = np.correlate(seg, template, mode='full')
        shift = np.argmax(corr) - (len(seg) - 1)
        new_r = r + shift
        new_r = max(0, min(len(signal) - 1, int(new_r)))
        refined.append(new_r)
    # deduplicate and sort
    refined = np.array(sorted(set(refined)), dtype=int)
    return refined


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    try:
        peaks = detect_r_peaks_pan_tompkin(signal, fs)
        # attempt to refine with template matching
        peaks_ref = refine_peaks_with_template(signal, peaks, fs)
        if len(peaks_ref) >= 1:
            return peaks_ref
        return peaks
    except Exception:
        # fallback simple detector
        filtered = bandpass(signal, fs)
        distance = int(0.2 * fs)  # at least 200ms between peaks
        peaks, _ = find_peaks(filtered, distance=distance, height=np.std(filtered) * 0.5)
        return peaks


def detect_pqrst(signal: np.ndarray, r_peaks: np.ndarray, fs: float) -> Dict[str, List[Optional[int]]]:
    """Detect P, Q, R, S, T sample indices around each R-peak.
    Returns dict with lists of indices (or None when not found).
    """
    n = len(signal)
    P_idx = []
    Q_idx = []
    R_idx = r_peaks.tolist()
    S_idx = []
    T_idx = []

    # Use a lightly filtered signal for P/T detection (wider band)
    filt_pt = bandpass(signal, fs, low=0.5, high=30.0)

    for i, r in enumerate(r_peaks):
        # Q: search 40ms before R for local minimum
        q_lo = max(0, r - int(0.04 * fs))
        q_hi = r
        q_window = filt_pt[q_lo:q_hi + 1]
        if q_window.size > 0:
            q_rel = int(np.argmin(q_window))
            q = q_lo + q_rel
        else:
            q = None
        # S: search 40ms after R for local minimum
        s_lo = r
        s_hi = min(n - 1, r + int(0.04 * fs))
        s_window = filt_pt[s_lo:s_hi + 1]
        if s_window.size > 0:
            s_rel = int(np.argmin(s_window))
            s = s_lo + s_rel
        else:
            s = None

        # P: search between (r - 0.2s) and Q for a local maximum (if Q exists)
        p_lo = max(0, r - int(0.25 * fs))
        p_hi = q - 1 if q is not None else r - int(0.05 * fs)
        if p_hi > p_lo:
            p_window = filt_pt[p_lo:p_hi + 1]
            if p_window.size > 0:
                p_rel = int(np.argmax(p_window))
                p = p_lo + p_rel
            else:
                p = None
        else:
            p = None

        # T: search between S and (r + 0.4s) for a local maximum
        t_lo = s + 1 if (s is not None) else r + int(0.02 * fs)
        t_hi = min(n - 1, r + int(0.45 * fs))
        if t_hi > t_lo:
            t_window = filt_pt[t_lo:t_hi + 1]
            if t_window.size > 0:
                t_rel = int(np.argmax(t_window))
                t = t_lo + t_rel
            else:
                t = None
        else:
            t = None

        P_idx.append(int(p) if p is not None else None)
        Q_idx.append(int(q) if q is not None else None)
        S_idx.append(int(s) if s is not None else None)
        T_idx.append(int(t) if t is not None else None)

    return {"P": P_idx, "Q": Q_idx, "R": R_idx, "S": S_idx, "T": T_idx}


def extract_features_from_signal(signal: np.ndarray, seconds_per_pixel: float) -> Dict[str, Any]:
    fs = 1.0 / seconds_per_pixel
    r_peaks = detect_r_peaks(signal, fs)
    times = r_peaks * seconds_per_pixel
    rr_intervals = np.diff(times)
    heart_rate = None
    if rr_intervals.size > 0:
        heart_rate = float(60.0 / np.mean(rr_intervals))

    pqrst = detect_pqrst(signal, r_peaks, fs)

    # compute intervals/durations where possible
    pr_intervals = []
    qrs_durations = []
    qt_intervals = []
    for i in range(len(r_peaks)):
        p = pqrst['P'][i]
        q = pqrst['Q'][i]
        r = pqrst['R'][i]
        s = pqrst['S'][i]
        t = pqrst['T'][i]
        if p is not None and r is not None:
            pr_intervals.append((r - p) * seconds_per_pixel)
        else:
            pr_intervals.append(None)
        if q is not None and s is not None:
            qrs_durations.append((s - q) * seconds_per_pixel)
        else:
            qrs_durations.append(None)
        if q is not None and t is not None:
            qt_intervals.append((t - q) * seconds_per_pixel)
        else:
            qt_intervals.append(None)

    results = {
        "r_peaks_indices": r_peaks.tolist(),
        "r_peaks_times": times.tolist(),
        "rr_intervals_s": rr_intervals.tolist(),
        "heart_rate_bpm": heart_rate,
        "pqrst_indices": pqrst,
        "pr_intervals_s": pr_intervals,
        "qrs_durations_s": qrs_durations,
        "qt_intervals_s": qt_intervals,
    }
    return results


def export_results(results: Dict[str, Any], out_path: str, fmt: str = "csv") -> None:
    ext = fmt.lower()
    if ext == "json":
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    else:
        # csv
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in results.items():
                writer.writerow([k, json.dumps(v)])


def plot_results(img: np.ndarray, signal: np.ndarray, r_peaks: List[int], seconds_per_pixel: float) -> None:
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Original ECG Image')
    plt.subplot(2, 1, 2)
    t = np.arange(len(signal)) * seconds_per_pixel
    plt.plot(t, signal, '-k')
    plt.plot(np.array(r_peaks) * seconds_per_pixel, signal[r_peaks], 'ro')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized amplitude')
    plt.tight_layout()
    plt.show()


def process_file(path: str, output: Optional[str] = None, out_format: str = 'csv', show_plot: bool = False) -> Dict[str, Any]:
    img = load_image(path)
    ok, msg = validate_image_quality(img)
    if not ok:
        raise ValueError(f"Image validation failed: {msg}")
    img = deskew_image(img)
    img = enhance_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cleaned = remove_gridlines(gray)
    signal, seconds_per_pixel = extract_waveform(cleaned)
    features = extract_features_from_signal(signal, seconds_per_pixel)
    results = {"source": os.path.basename(path), "features": features}
    if output:
        export_results(results, output, fmt=out_format)
    if show_plot:
        plot_results(img, signal, features.get("r_peaks_indices", []), seconds_per_pixel)
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simple ECG image digitizer and analyzer')
    parser.add_argument('--input', '-i', required=True, help='Input ECG image (PNG/JPG/PDF)')
    parser.add_argument('--output', '-o', help='Output results file (CSV or JSON)')
    parser.add_argument('--format', '-f', choices=['csv', 'json'], default='csv')
    parser.add_argument('--show-plot', action='store_true')
    args = parser.parse_args()
    res = process_file(args.input, args.output, args.format, args.show_plot)
    print(json.dumps(res, indent=2))
