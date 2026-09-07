"""Dependency-light binary ranking metrics used by every target model."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def roc_curve(positive_scores: Iterable[float], negative_scores: Iterable[float]) -> dict[str, np.ndarray]:
    """Return a tie-aware ROC curve with an explicit origin point."""

    positives = _vector(positive_scores, "positive_scores")
    negatives = _vector(negative_scores, "negative_scores")
    scores = np.concatenate((positives, negatives))
    labels = np.concatenate((np.ones(positives.size, dtype=np.int8), np.zeros(negatives.size, dtype=np.int8)))
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]
    distinct = np.r_[np.flatnonzero(np.diff(scores)), scores.size - 1]
    tp = np.cumsum(labels, dtype=np.int64)[distinct]
    fp = (distinct + 1) - tp
    return {
        "fpr": np.r_[0.0, fp / negatives.size],
        "tpr": np.r_[0.0, tp / positives.size],
        "thresholds": np.r_[np.inf, scores[distinct]],
    }


def pr_curve(positive_scores: Iterable[float], negative_scores: Iterable[float]) -> dict[str, np.ndarray]:
    positives = _vector(positive_scores, "positive_scores")
    negatives = _vector(negative_scores, "negative_scores")
    curve = roc_curve(positives, negatives)
    tp = curve["tpr"] * positives.size
    fp = curve["fpr"] * negatives.size
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    return {"precision": precision, "recall": curve["tpr"], "thresholds": curve["thresholds"]}


def area_under_curve(x: Iterable[float], y: Iterable[float]) -> float:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    order = np.argsort(x_values, kind="mergesort")
    return float(np.trapz(y_values[order], x_values[order]))


def threshold_at_fpr(negative_scores: Iterable[float], target_fpr: float) -> dict[str, float | int | bool]:
    """Match Work1: use the first descending negative-score rank reaching target FPR."""

    negatives = _vector(negative_scores, "negative_scores")
    if not 0.0 < target_fpr <= 1.0:
        raise ValueError("target_fpr must be in (0, 1]")
    rank = max(1, int(math.ceil(target_fpr * negatives.size)))
    threshold = float(np.sort(negatives)[::-1][rank - 1])
    false_positives = int(np.count_nonzero(negatives >= threshold))
    return {
        "target_fpr": float(target_fpr),
        "actual_fpr": float(false_positives / negatives.size),
        "threshold": threshold,
        "negative_count": int(negatives.size),
        "minimum_resolvable_fpr": float(1.0 / negatives.size),
        "resolvable": bool(negatives.size >= math.ceil(1.0 / target_fpr)),
    }


def confusion_at_threshold(
    positive_scores: Iterable[float], negative_scores: Iterable[float], threshold: float
) -> dict[str, float | int]:
    positives = _vector(positive_scores, "positive_scores")
    negatives = _vector(negative_scores, "negative_scores")
    tp = int(np.count_nonzero(positives >= threshold))
    fn = int(positives.size - tp)
    fp = int(np.count_nonzero(negatives >= threshold))
    tn = int(negatives.size - fp)
    precision = tp / max(1, tp + fp)
    recall = tp / positives.size
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(1e-15, precision + recall)),
        "fpr": float(fp / negatives.size),
        "tpr": float(recall),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_scores(
    positive_scores: Iterable[float],
    negative_scores: Iterable[float],
    fpr_targets: Iterable[float] = (1e-3, 1e-4),
) -> tuple[dict, dict[str, np.ndarray]]:
    positives = _vector(positive_scores, "positive_scores")
    negatives = _vector(negative_scores, "negative_scores")
    roc = roc_curve(positives, negatives)
    pr = pr_curve(positives, negatives)
    operating_points = {}
    for target in fpr_targets:
        calibration = threshold_at_fpr(negatives, float(target))
        key = f"fpr_{float(target):.0e}"
        operating_points[key] = {
            **calibration,
            **confusion_at_threshold(positives, negatives, float(calibration["threshold"])),
        }
    summary = {
        "positive_count": int(positives.size),
        "negative_count": int(negatives.size),
        "roc_auc": area_under_curve(roc["fpr"], roc["tpr"]),
        "average_precision": area_under_curve(pr["recall"], pr["precision"]),
        "positive_score": _distribution(positives),
        "negative_score": _distribution(negatives),
        "operating_points": operating_points,
    }
    curves = {**roc, "precision": pr["precision"], "recall": pr["recall"]}
    return summary, curves


def bootstrap_auc_ci(
    positive_scores: Iterable[float],
    negative_scores: Iterable[float],
    *,
    repetitions: int = 1000,
    confidence: float = 0.95,
    seed: int = 2025,
) -> dict[str, float | int]:
    """Stratified bootstrap confidence interval for ROC AUC."""

    positives = _vector(positive_scores, "positive_scores")
    negatives = _vector(negative_scores, "negative_scores")
    if repetitions <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("repetitions must be positive and confidence must be in (0, 1)")
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        pos = positives[generator.integers(0, positives.size, positives.size)]
        neg = negatives[generator.integers(0, negatives.size, negatives.size)]
        curve = roc_curve(pos, neg)
        estimates[index] = area_under_curve(curve["fpr"], curve["tpr"])
    alpha = (1.0 - confidence) / 2.0
    return {
        "confidence": float(confidence),
        "repetitions": int(repetitions),
        "low": float(np.quantile(estimates, alpha)),
        "high": float(np.quantile(estimates, 1.0 - alpha)),
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }
