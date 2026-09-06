"""从单一配置文件启动 Work2 训练或评估。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "vista_augur/configs/second_workpoint_deepcorr300_work1_aligned.jsonc"


def main() -> int:
    config_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_CONFIG
    sys.path.insert(0, str(PROJECT_ROOT / "vista_augur"))

    from second_workpoint.config import load_config

    config = load_config(config_path)
    python_executable = config.python_executable or sys.executable
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = config.visible_gpu_devices
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "vista_augur")

    command = [
        python_executable,
        str(PROJECT_ROOT / "vista_augur/run_train.py"),
        "--config",
        str(config_path),
    ]
    print(f"[launch] mode={config.run_mode} GPUs={config.visible_gpu_devices}")
    print(f"[launch] config={config_path}")
    return subprocess.call(command, cwd=PROJECT_ROOT, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
