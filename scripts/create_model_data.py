"""Build the shared operational datasets after STEP 02."""
import argparse
from pathlib import Path
from merry_llama.make_dataset import run_model_data, validate_model_data


def main():
    parser = argparse.ArgumentParser(description='PRC 2026 operational model datasets')
    parser.add_argument('--root', type=Path, help='Project root; defaults to installed checkout')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--overwrite', action='store_true', help='Rebuild STEP 03 outputs only')
    action.add_argument('--verify', action='store_true', help='Verify existing STEP 03 without building')
    args = parser.parse_args()
    if args.verify:
        validate_model_data(args.root)
        print('VERIFIED: operational model datasets (26 files, 69 added columns).', flush=True)
    else:
        run_model_data(args.root, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
