import os
import numpy as np
from PIL import Image, ImageDraw

import cv2
from ecg_analyzer.extractor import process_file, detect_r_peaks, detect_pqrst


def create_synthetic_ecg_image(path: str, width: int = 1200, height: int = 400):
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    # draw light grid
    for x in range(0, width, 25):
        draw.line([(x, 0), (x, height)], fill=(230, 230, 230))
    for y in range(0, height, 25):
        draw.line([(0, y), (width, y)], fill=(230, 230, 230))
    # draw a synthetic ECG trace
    t = np.linspace(0, 10, width)
    signal = 0.5 * np.sin(2 * np.pi * 1.2 * t)  # slow base
    # add peaks
    for center in np.arange(0.2, 10, 1.0):
        idx = int(center / 10 * width)
        for i in range(-3, 4):
            x = idx + i
            if 0 <= x < width:
                y = int(height / 2 - (signal[x] * 50) - (20 * np.exp(-i * i)))
                draw.point((x, y), fill='black')
    img.save(path)


def test_process_synthetic(tmp_path):
    p = tmp_path / "synthetic.png"
    create_synthetic_ecg_image(str(p))
    out = tmp_path / "out.json"
    res = process_file(str(p), str(out), 'json', show_plot=False)
    assert 'features' in res
    assert 'r_peaks_indices' in res['features']
    # additionally check detection routines on extracted signal
    # load and extract waveform via internal helper
    from ecg_analyzer.extractor import load_image, extract_waveform
    img = load_image(str(p))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    signal, seconds_per_pixel = extract_waveform(gray)
    r = detect_r_peaks(signal, 1.0 / seconds_per_pixel)
    pqrst = detect_pqrst(signal, r, 1.0 / seconds_per_pixel)
    assert len(pqrst['R']) == len(r)
