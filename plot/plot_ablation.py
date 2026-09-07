"""Ablation ROC entry point; conditions are read from standardized summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot.plot_roc import plot_roc


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot component or mechanism ablation ROC curves.")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("plot/results/ablation.pdf"))
    args = parser.parse_args()
    plot_roc(args.summaries, args.output, title="Ablation study")


if __name__ == "__main__":
    main()

