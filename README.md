# ECG Analyzer

Simple Python tool to digitize ECG graph images (PNG/JPG/PDF), extract waveform features (R-peaks, RR intervals, heart rate), and export results as CSV or JSON.

Quick start

1. Create a Python 3.8+ virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the CLI:

```powershell
python -m ecg_analyzer.cli --input path/to/ecg.png --output results.csv --format csv --show-plot
```

Notes
- PDF input requires `pdf2image` and a system `poppler` installation.
- This is a minimal reference implementation aimed at the features described in the SRS. Further validation, UX polish, and clinical validation are recommended before clinical use.
