"""Plot train/evaluation dynamics from one or more train_summary.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot.common import pyplot, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss, attack efficacy, and overhead over epochs.")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("plot/results/training_dynamics.pdf"))
    args = parser.parse_args()
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    for path in args.summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        epochs = data["epochs"]
        label = data.get("setting_name", path.parent.name)
        x = [item["epoch"] for item in epochs]
        axes[0].plot(x, [item["eval"]["loss"] for item in epochs], label=label)
        axes[1].plot(x, [item["eval"]["adv_positive_rate"] for item in epochs], label=label)
        axes[2].plot(x, [item["eval"]["time_ratio"] for item in epochs], label=f"{label} time")
        axes[2].plot(x, [item["eval"]["size_ratio"] for item in epochs], linestyle="--", label=f"{label} size")
    for axis, title, ylabel in zip(axes, ("Objective", "Attack response", "Perturbation cost"), ("Evaluation loss", "Adversarial positive rate", "Relative L2")):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()

