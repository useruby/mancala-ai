"""Publish the append-only seed477 counterfactual accounting correction."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.alphazero_lite.seed477_counterfactual_accounting import ROOT, publish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    publish(args.root)
    print("seed477 counterfactual accounting correction published")


if __name__ == "__main__":
    main()
