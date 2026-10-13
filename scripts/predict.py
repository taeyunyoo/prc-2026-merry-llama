"""Saved-model inference; never refits a model or learns from evaluation targets."""
import argparse
from pathlib import Path
from merry_llama.training import predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path)
    p.add_argument('--run-id',default='v55_reproduction')
    p.add_argument('--dataset',choices=['ranking','final'],default='final')
    # Older external launch cells may still pass this now-unnecessary option.
    p.add_argument('--accept-ranking-differences',action='store_true',help=argparse.SUPPRESS)
    args=vars(p.parse_args())
    args.pop('accept_ranking_differences')
    return predict(**args)


if __name__=='__main__':main()
