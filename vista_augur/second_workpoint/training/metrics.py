"""Metric helpers for fixed-threshold and low-FPR evaluation."""

from __future__ import annotations

import math

import numpy as np


def threshold_at_target_fpr(negative_scores, target_fpr: float) -> dict[str, float | int | bool]:
    """Match Work1 by selecting the first descending-score point reaching the target FPR."""

    scores = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("At least one negative score is required.")
    if not 0.0 < target_fpr <= 1.0:
        raise ValueError("target_fpr must be in (0, 1].")

    target_false_positive_rank = max(1, int(math.ceil(target_fpr * scores.size)))
    descending = np.sort(scores)[::-1]
    threshold = float(descending[min(target_false_positive_rank, scores.size) - 1])

    actual_false_positives = int(np.count_nonzero(scores >= threshold))
    return {
        "threshold": threshold,
        "target_fpr": float(target_fpr),
        "calibration_negative_count": int(scores.size),
        "target_false_positive_rank": target_false_positive_rank,
        "actual_calibration_fpr": float(actual_false_positives / scores.size),
        "minimum_resolvable_fpr": float(1.0 / scores.size),
        "resolvable": bool(scores.size >= math.ceil(1.0 / target_fpr)),
    }


def summarize_scores(positive_scores, negative_scores, threshold: float) -> dict[str, float | int]:
    positives = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negatives = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    tp = int(np.count_nonzero(positives >= threshold))
    fn = int(positives.size - tp)
    fp = int(np.count_nonzero(negatives >= threshold))
    tn = int(negatives.size - fp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(1e-12, precision + recall)),
        "fpr": float(fp / max(1, fp + tn)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
