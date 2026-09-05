"""Metric helpers for fixed-threshold and low-FPR evaluation."""

from __future__ import annotations

import math

import numpy as np


def threshold_at_target_fpr(negative_scores, target_fpr: float) -> dict[str, float | int | bool]:
    """Choose the lowest threshold whose empirical FPR does not exceed the target."""

    scores = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("At least one negative score is required.")
    if not 0.0 < target_fpr <= 1.0:
        raise ValueError("target_fpr must be in (0, 1].")

    allowed_false_positives = int(math.floor(target_fpr * scores.size))
    descending = np.sort(scores)[::-1]
    if allowed_false_positives <= 0:
        threshold = float(np.nextafter(descending[0], np.inf))
    elif allowed_false_positives >= scores.size:
        threshold = float(np.nextafter(descending[-1], -np.inf))
    else:
        threshold = float(descending[allowed_false_positives - 1])
        if int(np.count_nonzero(scores >= threshold)) > allowed_false_positives:
            threshold = float(np.nextafter(threshold, np.inf))

    actual_false_positives = int(np.count_nonzero(scores >= threshold))
    return {
        "threshold": threshold,
        "target_fpr": float(target_fpr),
        "calibration_negative_count": int(scores.size),
        "allowed_false_positives": allowed_false_positives,
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
