"""Manifest-driven evaluation entry point for all paper experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.artifacts import correlation_conditions, load_arrays, load_score_conditions
from data.metrics import bootstrap_auc_ci, confusion_at_threshold, evaluate_scores
from data.overhead import evaluate_overhead


def evaluate_experiment(spec: dict, base_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    artifact = (base_dir / spec["artifact"]).resolve()
    if spec.get("format") == "correlation_matrix":
        conditions = correlation_conditions(
            artifact,
            spec.get("matrix_key", "corr_matrix"),
            n_windows=int(spec.get("n_windows", 1)),
            vote_threshold=int(spec.get("vote_threshold", 1)),
        )
    else:
        conditions = load_score_conditions(artifact, spec.get("condition", "adversarial"))
    targets = tuple(float(value) for value in spec.get("fpr_targets", (1e-3, 1e-4)))
    summary = {
        "schema_version": 1,
        "name": spec["name"],
        "target": spec.get("target", "unknown"),
        "method": spec.get("method", spec["name"]),
        "category": spec.get("category", "work2"),
        "source": str(artifact),
        "conditions": {},
    }
    curve_arrays = {}
    raw = {}
    for condition, (positive, negative) in conditions.items():
        metrics, curves = evaluate_scores(positive, negative, targets)
        if int(spec.get("bootstrap", 0)) > 0:
            metrics["roc_auc_ci"] = bootstrap_auc_ci(
                positive,
                negative,
                repetitions=int(spec["bootstrap"]),
                confidence=float(spec.get("confidence", 0.95)),
                seed=int(spec.get("seed", 2025)),
            )
        summary["conditions"][condition] = metrics
        for name, values in curves.items():
            curve_arrays[f"{condition}__{name}"] = values
        raw[condition] = (positive, negative)
    if "clean" in raw:
        clean_thresholds = summary["conditions"]["clean"]["operating_points"]
        for condition, (positive, negative) in raw.items():
            if condition == "clean":
                continue
            fixed = {}
            for key, point in clean_thresholds.items():
                fixed[key] = {
                    "threshold_source": "clean",
                    "threshold": point["threshold"],
                    **confusion_at_threshold(positive, negative, float(point["threshold"])),
                }
            summary["conditions"][condition]["fixed_clean_threshold"] = fixed
    if spec.get("overhead_artifact"):
        arrays = load_arrays((base_dir / spec["overhead_artifact"]).resolve())
        summary["overhead"] = evaluate_overhead(
            arrays[spec.get("original_key", "original")],
            arrays[spec.get("adversarial_key", "adversarial")],
            time_channels=spec["time_channels"],
            size_channels=spec["size_channels"],
            valid_mask=arrays.get(spec.get("mask_key", "valid_mask")),
            time_unit_to_ms=float(spec.get("time_unit_to_ms", 1.0)),
            size_unit_to_kb=float(spec.get("size_unit_to_kb", 1.0)),
        )
    return summary, curve_arrays


def run_manifest(manifest_path: Path, output_dir: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = (manifest_path.parent / manifest.get("base_dir", ".")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for spec in manifest["experiments"]:
        summary, curves = evaluate_experiment(spec, base_dir)
        stem = _safe_name(spec["name"])
        curves_path = output_dir / f"{stem}.curves.npz"
        np.savez_compressed(curves_path, **curve_arrays(curves))
        summary["curves"] = curves_path.name
        summary_path = output_dir / f"{stem}.summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        index.append({"name": spec["name"], "summary": summary_path.name, "curves": curves_path.name})
    (output_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index


def curve_arrays(curves: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in curves.items()}


def _safe_name(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Work2 and compatible Work1 result artifacts.")
    parser.add_argument("--manifest", required=True, type=Path, help="JSON experiment manifest.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"), help="Evaluation output directory.")
    args = parser.parse_args()
    records = run_manifest(args.manifest.resolve(), args.output_dir.resolve())
    print(f"evaluated {len(records)} experiment(s); results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
