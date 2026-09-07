"""Plot latency distributions and resource summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plot.common import pyplot, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot efficiency JSON reports.")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--metric", default="latency_ms")
    parser.add_argument("--output", type=Path, default=Path("plot/results/efficiency.pdf"))
    args = parser.parse_args()
    rows = []
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if args.metric in report:
            rows.append((path.stem, report[args.metric]))
    if not rows:
        raise ValueError(f"metric {args.metric} was not found")
    x = np.arange(len(rows), dtype=np.float64)
    plt = pyplot()
    fig, axis = plt.subplots(figsize=(max(5.0, len(rows) * 1.2), 3.5))
    mean_artist = axis.bar(x, [row[1]["mean"] for row in rows], color="#4c78a8", label="Mean")
    p95_artist = axis.scatter(x, [row[1]["p95"] for row in rows], color="#d62728", marker="D", label="P95", zorder=3)
    p99_artist = axis.scatter(x, [row[1]["p99"] for row in rows], color="#8b0000", marker="x", label="P99", zorder=3)
    rotation = 0 if len(rows) <= 4 else 20
    axis.set_xticks(x, [row[0] for row in rows], rotation=rotation, ha="center" if rotation == 0 else "right")
    axis.set_ylabel(args.metric.replace("_", " ").title())
    axis.grid(axis="y", alpha=0.2)
    axis.margins(y=0.14)
    axis.legend(
        [mean_artist, p95_artist, p99_artist],
        ["Mean", "P95", "P99"],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncols=3,
    )
    fig.tight_layout()
    save_figure(fig, args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
