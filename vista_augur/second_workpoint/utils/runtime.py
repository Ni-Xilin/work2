"""运行时工具模块。

这里保留真实训练主线需要的通用动作：
- 随机种子
- 设备与 dtype 解析
- JSON 落盘
- 参数统计
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from second_workpoint.config import ExperimentConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: ExperimentConfig) -> str:
    return (
        f"torch_real(trainable={config.trainable_device}, "
        f"backbone={config.backbone_device}, target={config.target_device}, "
        f"dtype={config.backbone_dtype}, quant={config.backbone_quantization}, "
        f"semantic={config.semantic_alignment_mode})"
    )


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("当前操作需要 PyTorch，请在 deepcorr 环境中运行。") from exc
    return torch


def resolve_torch_device(device_name: str):
    torch = import_torch()
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"请求使用 {device_name}，但当前环境没有可用 CUDA。")
    return torch.device(device_name)


def resolve_torch_dtype(dtype_name: str):
    torch = import_torch()
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"不支持的 torch dtype: {dtype_name}")
    return mapping[dtype_name]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def save_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(_to_python(payload), file, ensure_ascii=False, indent=2)


def count_parameters(module: Any) -> tuple[int, int]:
    if hasattr(module, "parameters"):
        total = 0
        trainable = 0
        for parameter in module.parameters():
            size = int(parameter.numel())
            total += size
            if bool(getattr(parameter, "requires_grad", False)):
                trainable += size
        return total, trainable
    raise TypeError("module 必须实现 parameters() 才能统计参数。")


def _to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_python(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
