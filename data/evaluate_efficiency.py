"""Aggregate inference latency, throughput, CPU, memory, and training-time samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(values) -> dict[str, float | int]:
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("efficiency samples must be finite and non-empty")
    return {
        "count": int(samples.size),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "median": float(np.median(samples)),
        "p90": float(np.percentile(samples, 90)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Work2 efficiency measurements from NPZ arrays.")
    parser.add_argument("--input", required=True, type=Path, help="NPZ containing measurement arrays.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path.")
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as loaded:
        report = {key: summarize(loaded[key]) for key in loaded.files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"efficiency report: {args.output}")


if __name__ == "__main__":
    main()

