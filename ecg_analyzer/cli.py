"""Command-line interface for processing ECG images.

This module exposes a simple CLI entrypoint used by `python -m ecg_analyzer.cli`.
It supports single-file and batch processing and mirrors the options described
in the SRS: `--input`, `--output`, `--format`, and a `--show-plot` flag.
"""

from __future__ import annotations
import argparse
from .extractor import process_file


def main():
    """Parse arguments and run processing on one or more input files.

    - If multiple inputs are provided and `--output` is a folder, each file
      is written to `<out_folder>/<basename>_results.<format>`.
    - For a single input, `--output` may be a file path or omitted to print
      results to stdout as JSON.
    """
    parser = argparse.ArgumentParser(prog='ecg_extractor', description='ECG image digitizer and analyzer')
    parser.add_argument('--input', '-i', required=True, nargs='+', help='Input ECG image(s)')
    parser.add_argument('--output', '-o', help='Output results file or folder (for batch)')
    parser.add_argument('--format', '-f', choices=['csv', 'json'], default='csv')
    parser.add_argument('--show-plot', action='store_true')
    args = parser.parse_args()

    inputs = args.input
    # Batch case: multiple inputs and an output folder
    if len(inputs) > 1 and args.output and not args.output.endswith(('.csv', '.json')):
        out_folder = args.output or '.'
        for p in inputs:
            name = p.split('/')[-1]
            base = name.rsplit('.', 1)[0]
            out_path = f"{out_folder}/{base}_results.{args.format}"
            res = process_file(p, out_path, args.format, args.show_plot)
            print(f"Processed {p} -> {out_path}")
    else:
        # Single file: process and optionally write to the given output path
        p = inputs[0]
        out_path = args.output
        res = process_file(p, out_path, args.format, args.show_plot)
        if out_path:
            print(f"Wrote results to {out_path}")
        else:
            print(res)


if __name__ == '__main__':
    main()
