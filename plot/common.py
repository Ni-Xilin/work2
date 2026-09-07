"""Shared loading, styling, and output helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


COLORS = {
    "No-Perturbation": "#1f77b4",
    "Clean": "#1f77b4",
    "Work2": "#d62728",
    "Augur": "#8b0000",
    "Time": "#2ca02c",
    "Size": "#9467bd",
    "Both": "#d62728",
    "Adaptive": "#ff7f0e",
    "Transfer": "#17becf",
}


def pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires matplotlib; install the project requirements") from exc
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    return plt


def read_summary(path: str | Path) -> tuple[dict, Path]:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8")), source


def read_curves(summary: dict, summary_path: Path) -> dict[str, np.ndarray]:
    curve_path = summary_path.parent / summary["curves"]
    with np.load(curve_path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def save_figure(fig, output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight", transparent=destination.suffix.lower() in {".svg", ".pdf"})


def color_for(label: str, index: int) -> str:
    for key, color in COLORS.items():
        if key.lower() in label.lower():
            return color
    palette = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2")
    return palette[index % len(palette)]


def condition_label(summary: dict, condition: str) -> str:
    method = summary.get("method", summary.get("name", "method"))
    if condition.lower() in {"adv", "adversarial", "perturbed", "defended"}:
        return method
    if condition.lower() == "clean":
        return "No-Perturbation"
    return f"{method} - {condition.replace('_', ' ').title()}"
