"""Run metadata preprocessing after installing the project with pip install -e ."""
import argparse
from pathlib import Path

from merry_llama.preprocessor import run_preprocessing


def main():
    parser = argparse.ArgumentParser(description="PRC 2026 metadata preprocessing")
    parser.add_argument("--root", type=Path, help="Project root; defaults to the installed checkout")
    parser.add_argument("-s", "--set", choices=["train", "rank", "final", "all"],
                        default="all", dest="flight_set", help="Output datasets (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild derived metadata files")
    args = parser.parse_args()
    run_preprocessing(args.root, overwrite=args.overwrite, flight_set=args.flight_set)


if __name__ == "__main__":
    main()

