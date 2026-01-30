"""Example script to batch-process ECG files in a folder."""
import os
import argparse
from glob import glob
from ecg_analyzer.extractor import process_file


def main():
    parser = argparse.ArgumentParser(description='Batch process ECG images')
    parser.add_argument('input_folder')
    parser.add_argument('--out', '-o', default='results', help='Output folder')
    parser.add_argument('--format', '-f', choices=['csv', 'json'], default='csv')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.pdf']
    files = []
    for p in patterns:
        files.extend(glob(os.path.join(args.input_folder, p)))
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(args.out, f"{base}_results.{args.format}")
        try:
            process_file(f, out_path, args.format, show_plot=False)
            print(f"Processed {f} -> {out_path}")
        except Exception as e:
            print(f"Failed {f}: {e}")


if __name__ == '__main__':
    main()
