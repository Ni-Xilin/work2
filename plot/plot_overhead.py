"""Plot relative L2 and applied traffic overhead from evaluation summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot.common import pyplot, read_summary, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Work2 perturbation overhead.")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("plot/results/overhead.pdf"))
    args = parser.parse_args()
    rows = []
    for path in args.summaries:
        summary, _ = read_summary(path)
        if "overhead" in summary:
            rows.append((summary.get("target", summary["name"]), summary["overhead"]))
    if not rows:
        raise ValueError("none of the summaries contains overhead data")
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.2
    series = (
        ("Time L2", [row[1]["relative_l2"]["time"] for row in rows]),
        ("Size L2", [row[1]["relative_l2"]["size"] for row in rows]),
        ("Delay overhead", [row[1]["aggregate"]["relative_delay_overhead"] for row in rows]),
        ("Size overhead", [row[1]["aggregate"]["relative_size_overhead"] for row in rows]),
    )
    plt = pyplot()
    fig, axis = plt.subplots(figsize=(max(5.5, len(rows) * 1.3), 3.7))
    for index, (label, values) in enumerate(series):
        axis.bar(x + (index - 1.5) * width, values, width, label=label)
    axis.set_xticks(x, [row[0] for row in rows])
    axis.set_ylabel("Relative overhead")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncols=2)
    fig.tight_layout()
    save_figure(fig, args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()

