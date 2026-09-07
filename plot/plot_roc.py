"""Plot one or more target-model ROC panels from standardized summaries."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from plot.common import color_for, condition_label, pyplot, read_curves, read_summary, save_figure


def plot_roc(summary_paths: list[Path], output: Path, *, min_fpr: float = 1e-4, title: str = "") -> None:
    records = []
    for path in summary_paths:
        summary, source = read_summary(path)
        records.append((summary, read_curves(summary, source)))
    targets = list(dict.fromkeys(record[0].get("target", "unknown") for record in records))
    plt = pyplot()
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.5), squeeze=False, sharey=True)
    for target_index, target in enumerate(targets):
        axis = axes[0, target_index]
        line_index = 0
        for summary, curves in records:
            if summary.get("target", "unknown") != target:
                continue
            for condition in summary["conditions"]:
                fpr = curves[f"{condition}__fpr"]
                tpr = curves[f"{condition}__tpr"]
                label = condition_label(summary, condition)
                axis.plot(fpr, tpr, label=label, color=color_for(label, line_index), linewidth=1.7)
                line_index += 1
        axis.plot([min_fpr, 1.0], [min_fpr, 1.0], "--", color="0.65", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_xlim(min_fpr, 1.0)
        axis.set_ylim(0.0, 1.02)
        axis.set_title(target)
        axis.set_xlabel("False Positive Rate")
        axis.grid(True, which="both", alpha=0.2)
        axis.legend(frameon=False)
    axes[0, 0].set_ylabel("True Positive Rate")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper-ready ROC curves.")
    parser.add_argument("summaries", nargs="+", type=Path, help="Evaluation summary JSON files.")
    parser.add_argument("--output", type=Path, default=Path("plot/results/roc.pdf"))
    parser.add_argument("--min-fpr", type=float, default=1e-4)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    if args.min_fpr <= 0 or not math.isfinite(args.min_fpr):
        raise ValueError("min-fpr must be finite and positive")
    plot_roc(args.summaries, args.output, min_fpr=args.min_fpr, title=args.title)


if __name__ == "__main__":
    main()

