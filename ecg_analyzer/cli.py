from __future__ import annotations
import argparse
from .extractor import process_file


def main():
    parser = argparse.ArgumentParser(prog='ecg_extractor', description='ECG image digitizer and analyzer')
    parser.add_argument('--input', '-i', required=True, nargs='+', help='Input ECG image(s)')
    parser.add_argument('--output', '-o', help='Output results file or folder (for batch)')
    parser.add_argument('--format', '-f', choices=['csv', 'json'], default='csv')
    parser.add_argument('--show-plot', action='store_true')
    args = parser.parse_args()

    inputs = args.input
    if len(inputs) > 1 and args.output and not args.output.endswith(('.csv', '.json')):
        out_folder = args.output
        if not out_folder:
            out_folder = '.'
        for p in inputs:
            name = p.split('/')[-1]
            base = name.rsplit('.', 1)[0]
            out_path = f"{out_folder}/{base}_results.{args.format}"
            res = process_file(p, out_path, args.format, args.show_plot)
            print(f"Processed {p} -> {out_path}")
    else:
        p = inputs[0]
        out_path = args.output
        res = process_file(p, out_path, args.format, args.show_plot)
        if out_path:
            print(f"Wrote results to {out_path}")
        else:
            print(res)


if __name__ == '__main__':
    main()
