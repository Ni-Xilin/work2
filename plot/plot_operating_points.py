"""Plot precision, recall, and F1 at paper low-FPR operating points."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot.common import color_for, condition_label, pyplot, read_summary, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot low-FPR precision/recall/F1 grouped bars.")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--fpr", default="fpr_1e-04", help="Operating-point key from the summary.")
    parser.add_argument("--output", type=Path, default=Path("plot/results/operating_points.pdf"))
    args = parser.parse_args()
    rows = []
    for path in args.summaries:
        summary, _ = read_summary(path)
        for condition, metrics in summary["conditions"].items():
            if args.fpr in metrics["operating_points"]:
                rows.append((summary.get("target", "unknown"), condition_label(summary, condition), metrics["operating_points"][args.fpr]))
    if not rows:
        raise ValueError(f"no summaries contain operating point {args.fpr}")
    labels = [f"{target}\n{label}" for target, label, _ in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.24
    plt = pyplot()
    fig, axis = plt.subplots(figsize=(max(6.0, len(rows) * 1.1), 3.8))
    for offset, metric in zip((-width, 0.0, width), ("precision", "recall", "f1")):
        axis.bar(x + offset, [row[2][metric] for row in rows], width, label=metric.title(), color=color_for(metric, int((offset + width) / width)))
    axis.set_xticks(x, labels, rotation=0, ha="center", fontsize=8)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncols=3)
    fig.tight_layout()
    save_figure(fig, args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
