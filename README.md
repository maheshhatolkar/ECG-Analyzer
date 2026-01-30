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

Electron wrapper (bundle Flask UI)

You can wrap the Flask web UI in an Electron shell so the app runs as a desktop application with an embedded browser. The scaffold lives in `web/electron`.

1. Install Node.js (14+ recommended) and then install dev dependencies:

```powershell
cd web/electron
npm install
```

2. Start (this spawns the Flask server and opens the Electron window):

```powershell
npm start
```

Notes:
- The Electron `main.js` spawns `python web/app.py`. Ensure the intended Python interpreter is available as `python` in PATH, or set the `PYTHON` environment variable when launching Electron to point to your virtualenv python (e.g., `.\.venv\Scripts\python.exe`).
- For packaging the Electron app (creating an installer), you can use `electron-builder` or `electron-packager`; I can add CI packaging for cross-platform builds on request.


Packaging (PyInstaller)

You can create a standalone desktop executable for the Tkinter GUI with PyInstaller.

1. Install PyInstaller in your environment:

```powershell
pip install pyinstaller
```

2. Build (PowerShell script provided):

```powershell
# one-file build (may increase startup time)
.\scripts\build_pyinstaller.ps1 -onefile

# or specify a name and entry
.\scripts\build_pyinstaller.ps1 -entry run_gui.py -name ECGAnalyzerGUI -onefile
```

3. Output: `dist\<name>` will contain the executable.

Notes:
- The script adds `web/templates` into the bundle so the web UI can be packaged if desired.
- Packaging the Flask web app into a single exe requires bundling server startup and possibly switching to a production WSGI server; for desktop usage prefer the Tkinter GUI.
- For cross-platform packaging on macOS or Linux, run `pyinstaller` on the target OS or use CI runners.

