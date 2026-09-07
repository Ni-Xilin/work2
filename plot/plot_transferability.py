"""Transferability/adaptive-attack ROC entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot.plot_roc import plot_roc


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot white-box, black-box, or adaptive-attack ROC curves.")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("plot/results/transferability.pdf"))
    parser.add_argument("--title", default="Transferability and adaptive attack")
    args = parser.parse_args()
    plot_roc(args.summaries, args.output, title=args.title)


if __name__ == "__main__":
    main()

