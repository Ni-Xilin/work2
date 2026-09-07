"""Generate every configured paper figure from one JSON figure manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPTS = {
    "roc": "plot.plot_roc",
    "operating_points": "plot.plot_operating_points",
    "score_matrix": "plot.plot_score_matrix",
    "overhead": "plot.plot_overhead",
    "training": "plot.plot_training",
    "efficiency": "plot.plot_efficiency",
    "ablation": "plot.plot_ablation",
    "transferability": "plot.plot_transferability",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all figures declared in a JSON manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base = (args.manifest.parent / manifest.get("base_dir", ".")).resolve()
    for figure in manifest["figures"]:
        kind = figure["type"]
        if kind not in SCRIPTS:
            raise ValueError(f"unknown figure type: {kind}")
        command = [sys.executable, "-m", SCRIPTS[kind]]
        for value in figure.get("inputs", []):
            command.append(str((base / value).resolve()))
        for key, value in figure.get("options", {}).items():
            option = f"--{key.replace('_', '-')}"
            values = value if isinstance(value, list) else [value]
            command.append(option)
            command.extend(
                str((base / item).resolve()) if key in {"input", "output"} else str(item)
                for item in values
            )
        subprocess.run(command, check=True)
        print(f"generated {figure.get('name', kind)}")


if __name__ == "__main__":
    main()
