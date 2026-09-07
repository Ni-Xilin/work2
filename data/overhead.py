"""Physical-deployability and perturbation-overhead evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def evaluate_overhead(
    original,
    adversarial,
    *,
    time_channels: Iterable[int],
    size_channels: Iterable[int],
    valid_mask=None,
    time_unit_to_ms: float = 1.0,
    size_unit_to_kb: float = 1.0,
    tolerance: float = 1e-8,
) -> dict:
    """Evaluate additive delay/padding for arrays shaped (..., channels, length)."""

    clean = np.asarray(original, dtype=np.float64)
    defended = np.asarray(adversarial, dtype=np.float64)
    if clean.shape != defended.shape or clean.ndim < 2:
        raise ValueError("original and adversarial must have the same (..., channels, length) shape")
    delta = np.abs(defended) - np.abs(clean)
    if valid_mask is None:
        mask = np.ones_like(delta, dtype=bool)
    else:
        candidate_mask = np.asarray(valid_mask)
        if candidate_mask.shape == clean.shape[:-2] + (clean.shape[-1],):
            candidate_mask = np.expand_dims(candidate_mask, axis=-2)
        mask = np.broadcast_to(candidate_mask, delta.shape).astype(bool)
    time_idx = _indices(time_channels, clean.shape[-2], "time_channels")
    size_idx = _indices(size_channels, clean.shape[-2], "size_channels")
    time_delta = np.take(delta, time_idx, axis=-2)
    size_delta = np.take(delta, size_idx, axis=-2)
    time_mask = np.take(mask, time_idx, axis=-2)
    size_mask = np.take(mask, size_idx, axis=-2)
    time_clean = np.take(np.abs(clean), time_idx, axis=-2)
    size_clean = np.take(np.abs(clean), size_idx, axis=-2)
    return {
        "shape": list(clean.shape),
        "physical_constraints": {
            "time_negative_delta_count": int(np.count_nonzero((time_delta < -tolerance) & time_mask)),
            "size_negative_delta_count": int(np.count_nonzero((size_delta < -tolerance) & size_mask)),
            "direction_flip_count": int(np.count_nonzero((np.sign(clean) != np.sign(defended)) & mask & (np.abs(clean) > tolerance))),
            "finite": bool(np.all(np.isfinite(defended[mask]))),
        },
        "relative_l2": {
            "time": _relative_l2(time_delta[time_mask], time_clean[time_mask]),
            "size": _relative_l2(size_delta[size_mask], size_clean[size_mask]),
        },
        "delay": _additive_statistics(time_delta[time_mask] * time_unit_to_ms, "ms", tolerance),
        "padding": _additive_statistics(size_delta[size_mask] * size_unit_to_kb, "KB", tolerance),
        "aggregate": {
            "relative_delay_overhead": _relative_l1(time_delta[time_mask], time_clean[time_mask]),
            "relative_size_overhead": _relative_l1(size_delta[size_mask], size_clean[size_mask]),
        },
    }


def _indices(values: Iterable[int], channel_count: int, name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.int64)
    if result.size == 0 or np.any(result < 0) or np.any(result >= channel_count):
        raise ValueError(f"{name} must contain valid channel indices")
    return result


def _relative_l2(delta: np.ndarray, original: np.ndarray) -> float:
    return float(np.linalg.norm(delta) / max(1e-15, np.linalg.norm(original)))


def _relative_l1(delta: np.ndarray, original: np.ndarray) -> float:
    return float(np.sum(np.clip(delta, 0.0, None)) / max(1e-15, np.sum(np.abs(original))))


def _additive_statistics(delta: np.ndarray, unit: str, tolerance: float) -> dict:
    positive = delta[delta > tolerance]
    total = int(delta.size)
    result = {
        "unit": unit,
        "total_value_count": total,
        "modified_value_count": int(positive.size),
        "modified_fraction": float(positive.size / max(1, total)),
        "total_added": float(np.sum(np.clip(delta, 0.0, None))),
    }
    for name, value in (("mean", np.mean), ("median", np.median), ("max", np.max)):
        result[name] = float(value(positive)) if positive.size else 0.0
    for percentile in (90, 95, 99):
        result[f"p{percentile}"] = float(np.percentile(positive, percentile)) if positive.size else 0.0
    return result
