"""从单一配置文件启动 Work2 训练或评估。"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "vista_augur/configs/second_workpoint_deepcorr300_work1_aligned.jsonc"


@dataclass(frozen=True)
class GpuState:
    index: str
    uuid: str
    name: str
    total_mb: int
    used_mb: int
    utilization: int

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb


def _query_gpus() -> list[GpuState]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    states = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"无法解析 nvidia-smi 输出: {line}")
        states.append(GpuState(fields[0], fields[1], fields[2], int(fields[3]), int(fields[4]), int(fields[5])))
    return states


def _query_compute_processes() -> dict[str, list[str]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    processes: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) == 4:
            processes.setdefault(fields[0], []).append(
                f"pid={fields[1]} process={fields[2]} memory={fields[3]} MiB"
            )
    return processes


def _select_gpus(config) -> str:
    states = _query_gpus()
    processes = _query_compute_processes()
    print("[gpu-check] index | name | used/total MiB | utilization")
    for state in states:
        print(
            f"[gpu-check] {state.index} | {state.name} | "
            f"{state.used_mb}/{state.total_mb} | {state.utilization}%"
        )
        for process in processes.get(state.uuid, []):
            print(f"[gpu-process] GPU {state.index} | {process}")

    suitable = {
        state.index: state
        for state in states
        if state.used_mb <= config.gpu_max_memory_used_mb
        and state.utilization <= config.gpu_max_utilization_percent
        and state.free_mb >= config.gpu_min_free_memory_mb
        and not processes.get(state.uuid)
    }
    requested = config.visible_gpu_devices.strip().lower()
    if requested == "auto":
        selected = list(suitable)[: config.required_gpu_count]
    else:
        selected = [value.strip() for value in config.visible_gpu_devices.split(",") if value.strip()]

    if len(selected) != config.required_gpu_count or len(set(selected)) != len(selected):
        raise RuntimeError(
            f"必须选择 {config.required_gpu_count} 张不同的 GPU，当前选择: {selected}"
        )
    unavailable = [index for index in selected if index not in suitable]
    if unavailable:
        raise RuntimeError(
            f"GPU {unavailable} 不满足空闲阈值，已拒绝启动。"
            "请等待资源释放或在配置文件中选择其他 GPU。"
        )
    return ",".join(selected)


def main() -> int:
    config_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_CONFIG
    sys.path.insert(0, str(PROJECT_ROOT / "vista_augur"))

    from second_workpoint.config import load_config

    config = load_config(config_path)
    python_executable = config.python_executable or sys.executable
    environment = os.environ.copy()
    selected_gpus = config.visible_gpu_devices
    if config.gpu_preflight_enabled:
        selected_gpus = _select_gpus(config)
    if selected_gpus:
        environment["CUDA_VISIBLE_DEVICES"] = selected_gpus
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "vista_augur")

    command = [
        python_executable,
        str(PROJECT_ROOT / "vista_augur/run_train.py"),
        "--config",
        str(config_path),
    ]
    print(f"[launch] mode={config.run_mode} GPUs={selected_gpus or '保留当前环境'}")
    print(f"[launch] config={config_path}")
    return subprocess.call(command, cwd=PROJECT_ROOT, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
