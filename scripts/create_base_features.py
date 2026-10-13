"""Build stage 2 after installing this repository with python -m pip install -e ."""
import argparse
from pathlib import Path

from merry_llama.features.base import run_base_features, validate_base_outputs


def main():
    parser = argparse.ArgumentParser(description="PRC 2026: core and external features")
    parser.add_argument("--root", type=Path, help="Project root; defaults to installed checkout")
    parser.add_argument("--weather", type=Path, help="Explicit weather parquet (relative to root or absolute)")
    parser.add_argument("--reference", type=Path, help="Explicit reference CSV (relative to root or absolute)")
    parser.add_argument("--distance", type=Path, help="Explicit distance CSV (relative to root or absolute)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--overwrite", action="store_true", help="Rebuild base features only")
    action.add_argument("--verify", action="store_true", help="Verify saved inputs/outputs without building")
    args = parser.parse_args()
    if args.verify:
        if any(x is not None for x in (args.weather, args.reference, args.distance)):
            parser.error("--verify checks recorded sources; do not supply resource overrides")
        validate_base_outputs(args.root)
        print("VERIFIED: base/external feature generation (14 files, 46 added columns).", flush=True)
    else:
        run_base_features(args.root, weather_path=args.weather,
                          reference_path=args.reference, distance_path=args.distance,
                          overwrite=args.overwrite)


if __name__ == "__main__":
    main()
