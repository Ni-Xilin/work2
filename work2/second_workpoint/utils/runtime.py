"""运行时工具模块。
这个文件负责随机种子、运行设备描述、目录创建、数值算子和 JSON/NPZ 落盘等通用动作，
让训练器本身只关注训练流程。"""

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
    if config.backend == "numpy_stub":
        return "numpy(cpu-only stub)"
    if config.backend == "torch_real":
        return (
            f"torch_real(trainable={config.trainable_device}, "
            f"backbone={config.backbone_device}, target={config.target_device}, "
            f"dtype={config.backbone_dtype}, quant={config.backbone_quantization}, "
            f"semantic={config.semantic_alignment_mode})"
        )
    return "unknown"


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


def gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * np.power(values, 3))))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def layer_norm(values: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    mean = values.mean(axis=-1, keepdims=True)
    variance = np.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return ((values - mean) / np.sqrt(variance + epsilon)).astype(np.float32)


def save_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(_to_python(payload), file, ensure_ascii=False, indent=2)


def save_npz(path: str | Path, payload: dict[str, np.ndarray]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **{name: np.asarray(value) for name, value in payload.items()})


def count_parameters(module: Any) -> tuple[int, int]:
    if hasattr(module, "parameter_specs"):
        total = 0
        trainable = 0
        for _, value, requires_grad in module.parameter_specs():
            size = int(np.asarray(value).size)
            total += size
            if requires_grad:
                trainable += size
        return total, trainable
    if hasattr(module, "parameters"):
        total = 0
        trainable = 0
        for parameter in module.parameters():
            size = int(parameter.numel())
            total += size
            if bool(getattr(parameter, "requires_grad", False)):
                trainable += size
        return total, trainable
    raise TypeError("module 必须实现 parameter_specs() 才能统计参数。")


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
