"""Plot clean/defended correlation-score matrices with shared normalization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot.common import pyplot, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one or more square score matrices from NPZ.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("plot/results/score_matrices.pdf"))
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as loaded:
        matrices = [np.asarray(loaded[key], dtype=np.float64)[: args.max_items, : args.max_items] for key in args.keys]
    if any(matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] for matrix in matrices):
        raise ValueError("every selected matrix must be square")
    labels = args.labels or [key.replace("_", " ").title() for key in args.keys]
    if len(labels) != len(matrices):
        raise ValueError("labels and keys must have equal lengths")
    lower = min(float(np.min(matrix)) for matrix in matrices)
    upper = max(float(np.max(matrix)) for matrix in matrices)
    plt = pyplot()
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.5 * len(matrices), 3.2), squeeze=False)
    image = None
    for axis, matrix, label in zip(axes[0], matrices, labels):
        image = axis.imshow(matrix, cmap="viridis", vmin=lower, vmax=upper, interpolation="nearest", aspect="auto")
        axis.set_title(label)
        axis.set_xlabel("Exit flow")
        axis.set_ylabel("Ingress flow")
    fig.colorbar(image, ax=list(axes[0]), label="Correlation score", shrink=0.82)
    save_figure(fig, args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()

