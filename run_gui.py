"""Entry-point script for desktop packaging: launches the Tkinter GUI.

This tiny module is useful for PyInstaller and other packagers as a stable
entrypoint. It imports the GUI runner and invokes it. Keep this file minimal
so packaging tools can discover a single script to start the app.
"""
from ecg_analyzer.gui import run_gui


def main():
    # Launch the GUI application
    run_gui()


if __name__ == '__main__':
    main()
