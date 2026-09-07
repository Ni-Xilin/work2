"""Validate generated metric reports before values are copied into a paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_report(report: dict, tolerance: float = 1e-9) -> list[str]:
    errors = []
    for condition, metrics in report.get("conditions", {}).items():
        for key, point in metrics.get("operating_points", {}).items():
            expected = 2.0 * point["precision"] * point["recall"] / max(
                1e-15, point["precision"] + point["recall"]
            )
            if abs(expected - point["f1"]) > tolerance:
                errors.append(f"{condition}/{key}: inconsistent F1")
            if point["actual_fpr"] + tolerance < point["target_fpr"]:
                errors.append(f"{condition}/{key}: threshold did not reach target FPR")
            if not point["resolvable"]:
                errors.append(f"{condition}/{key}: target FPR is not empirically resolvable")
    overhead = report.get("overhead", {}).get("physical_constraints", {})
    for key in ("time_negative_delta_count", "size_negative_delta_count", "direction_flip_count"):
        if overhead.get(key, 0):
            errors.append(f"overhead/{key}: {overhead[key]} physical constraint violations")
    if overhead and not overhead.get("finite", True):
        errors.append("overhead: non-finite adversarial values")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate evaluation summaries before paper reporting.")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--allow-unresolvable", action="store_true", help="Allow unsupported low-FPR targets.")
    args = parser.parse_args()
    failed = False
    for path in args.summaries:
        report = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_report(report, args.tolerance)
        if args.allow_unresolvable:
            errors = [value for value in errors if "not empirically resolvable" not in value]
        if errors:
            failed = True
            print(f"FAIL {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
