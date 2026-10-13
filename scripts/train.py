"""Fit or reuse saved models, then generate the ranking submission."""
import argparse
from pathlib import Path
from merry_llama.training import train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path)
    p.add_argument('--run-id',default='v55_reproduction')
    p.add_argument('--threads',type=int,default=-1)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--allow-version-mismatch',action='store_true')
    return train(**vars(p.parse_args()))


if __name__=='__main__':main()
