"""Readers for Work2 bundles and legacy Work1 result files."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


SCORE_SUFFIXES = ("positive_scores", "negative_scores")


def load_score_conditions(path: str | Path, condition: str = "adversarial") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"result artifact does not exist: {source}")
    payload = _load(source)
    conditions = _standard_conditions(payload)
    if conditions:
        return conditions
    scores, labels = _legacy_scores(payload)
    return {condition: (scores[labels == 1], scores[labels == 0])}


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    payload = _load(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("array artifact must be a mapping")
    return {str(key): np.asarray(value) for key, value in payload.items()}


def correlation_conditions(
    path: str | Path,
    matrix_key: str = "corr_matrix",
    *,
    n_windows: int = 1,
    vote_threshold: int = 1,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arrays = load_arrays(path)
    matrices = {key: value for key, value in arrays.items() if key == matrix_key or key.endswith("_matrix")}
    if not matrices and matrix_key in arrays:
        matrices = {"adversarial": arrays[matrix_key]}
    result = {}
    for name, matrix in matrices.items():
        matrix = _session_matrix(
            np.asarray(matrix, dtype=np.float64),
            n_windows=n_windows,
            vote_threshold=vote_threshold,
        )
        diagonal = np.diag(matrix)
        result[name.removesuffix("_matrix")] = (diagonal, matrix[~np.eye(matrix.shape[0], dtype=bool)])
    if not result:
        raise ValueError(f"no square correlation matrix found in {path}")
    return result


def _load(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as loaded:
            return {key: loaded[key] for key in loaded.files}
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".p", ".pickle", ".pkl"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    raise ValueError(f"unsupported artifact type: {suffix}")


def _standard_conditions(payload) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not isinstance(payload, dict):
        return {}
    result = {}
    for key in payload:
        if not key.endswith("_positive_scores"):
            continue
        condition = key[: -len("_positive_scores")]
        negative_key = f"{condition}_negative_scores"
        if negative_key in payload:
            result[condition] = (np.asarray(payload[key]).reshape(-1), np.asarray(payload[negative_key]).reshape(-1))
    for condition, value in payload.items():
        if isinstance(value, dict) and "all_outputs" in value and "all_labels" in value:
            scores = np.asarray(value["all_outputs"]).reshape(-1)
            labels = np.asarray(value["all_labels"]).reshape(-1)
            result[str(condition)] = (scores[labels == 1], scores[labels == 0])
    return result


def _legacy_scores(payload) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(payload, dict) and "all_outputs" in payload and "all_labels" in payload:
        return np.asarray(payload["all_outputs"]).reshape(-1), np.asarray(payload["all_labels"]).reshape(-1)
    if isinstance(payload, (tuple, list)) and len(payload) >= 2:
        first = np.asarray(payload[0]).reshape(-1)
        second = np.asarray(payload[1]).reshape(-1)
        if first.size == second.size:
            if set(np.unique(second)).issubset({0, 1}):
                return first, second.astype(np.int8)
            if set(np.unique(first)).issubset({0, 1}):
                return second, first.astype(np.int8)
    raise ValueError("artifact does not contain a recognized score/label representation")


def _session_matrix(matrix: np.ndarray, *, n_windows: int, vote_threshold: int) -> np.ndarray:
    if n_windows <= 0 or not 1 <= vote_threshold <= n_windows:
        raise ValueError("vote_threshold must be between 1 and n_windows")
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        if n_windows == 1:
            return matrix
        if matrix.shape[0] % n_windows != 0:
            raise ValueError("legacy DeepCoFFEA matrix size must be divisible by n_windows")
        session_count = matrix.shape[0] // n_windows
        windows = np.stack(
            [
                matrix[index * session_count : (index + 1) * session_count, index * session_count : (index + 1) * session_count]
                for index in range(n_windows)
            ],
            axis=0,
        )
        # A pair receives a positive vote when its window score is >= eta.  The
        # vote decision is therefore equivalent to thresholding the kth-largest
        # window score, which gives one scalar session score for exact ROC use.
        return np.sort(windows, axis=0)[-vote_threshold]
    if matrix.ndim == 3 and matrix.shape[1] == matrix.shape[2]:
        if matrix.shape[0] == n_windows:
            return np.sort(matrix, axis=0)[-vote_threshold]
        return matrix.mean(axis=0)
    raise ValueError("correlation matrix must be square or a stack of square matrices")
