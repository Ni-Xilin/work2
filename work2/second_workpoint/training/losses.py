"""损失函数模块。
这个文件复用 Generator_Trainer 的思路：一部分 loss 负责压低目标模型置信，
另一部分 loss 负责约束时间与大小扰动的相对开销。"""

from __future__ import annotations

from typing import Any

import numpy as np

from second_workpoint.config import ExperimentConfig
from second_workpoint.utils.runtime import sigmoid


def _safe_l2_ratio(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        reference = reference * mask[:, None, :]
        candidate = candidate * mask[:, None, :]
    reference_flat = reference.reshape(reference.shape[0], -1)
    candidate_flat = candidate.reshape(candidate.shape[0], -1)
    distance = np.linalg.norm(candidate_flat - reference_flat, axis=1)
    reference_norm = np.linalg.norm(reference_flat, axis=1)
    reference_norm = np.maximum(reference_norm, 1e-6)
    return float(np.mean(distance / reference_norm))


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "device") and hasattr(value, "requires_grad")


def _coerce_torch_tensor(value: Any, reference_tensor):
    torch_module = _import_torch()
    if _is_torch_tensor(value):
        return value.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
    return torch_module.as_tensor(value, device=reference_tensor.device, dtype=reference_tensor.dtype)


def _safe_l2_ratio_torch(reference, candidate, mask=None):
    torch_module = _import_torch()
    if mask is not None:
        mask_tensor = _coerce_torch_tensor(mask, reference)
        reference = reference * mask_tensor[:, None, :]
        candidate = candidate * mask_tensor[:, None, :]
    reference_flat = reference.reshape(reference.shape[0], -1)
    candidate_flat = candidate.reshape(candidate.shape[0], -1)
    distance = torch_module.linalg.norm(candidate_flat - reference_flat, dim=1)
    reference_norm = torch_module.linalg.norm(reference_flat, dim=1).clamp_min(1e-6)
    return (distance / reference_norm).mean()


def _binary_cross_entropy_with_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = sigmoid(logits)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    loss = -(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
    return float(np.mean(loss))


def _binary_cross_entropy_with_logits_torch(logits, labels):
    torch_module = _import_torch()
    labels_tensor = _coerce_torch_tensor(labels, logits)
    return torch_module.nn.functional.binary_cross_entropy_with_logits(logits, labels_tensor)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Torch-native loss path requires PyTorch to be installed.") from exc
    return torch


class TargetedOverheadLoss:
    def __init__(self, config: ExperimentConfig) -> None:
        self.beta = config.beta
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.time_channel_indices = config.time_channel_indices
        self.size_channel_indices = config.size_channel_indices

    def forward(
        self,
        target_logits,
        target_labels,
        original_flow,
        adv_flow,
        flow_mask=None,
    ) -> dict[str, Any]:
        if any(_is_torch_tensor(value) for value in (target_logits, target_labels, original_flow, adv_flow, flow_mask)):
            return self._forward_torch(
                target_logits=target_logits,
                target_labels=target_labels,
                original_flow=original_flow,
                adv_flow=adv_flow,
                flow_mask=flow_mask,
            )

        label_loss = _binary_cross_entropy_with_logits(target_logits, target_labels)
        time_ratio = self._compute_ratio(original_flow, adv_flow, self.time_channel_indices, flow_mask)
        size_ratio = self._compute_ratio(original_flow, adv_flow, self.size_channel_indices, flow_mask)
        total_loss = self.beta * label_loss + self.alpha * time_ratio + self.gamma * size_ratio
        return {
            "loss": float(total_loss),
            "label_loss": float(label_loss),
            "time_ratio": float(time_ratio),
            "size_ratio": float(size_ratio),
        }

    def _forward_torch(
        self,
        target_logits,
        target_labels,
        original_flow,
        adv_flow,
        flow_mask=None,
    ):
        reference_tensor = next(
            value for value in (target_logits, original_flow, adv_flow, target_labels, flow_mask) if _is_torch_tensor(value)
        )
        logits = _coerce_torch_tensor(target_logits, reference_tensor)
        labels = _coerce_torch_tensor(target_labels, logits)
        reference_flow = _coerce_torch_tensor(original_flow, logits)
        candidate_flow = _coerce_torch_tensor(adv_flow, logits)
        mask_tensor = None if flow_mask is None else _coerce_torch_tensor(flow_mask, logits)

        label_loss = _binary_cross_entropy_with_logits_torch(logits, labels)
        time_ratio = self._compute_ratio_torch(reference_flow, candidate_flow, self.time_channel_indices, mask_tensor)
        size_ratio = self._compute_ratio_torch(reference_flow, candidate_flow, self.size_channel_indices, mask_tensor)
        total_loss = self.beta * label_loss + self.alpha * time_ratio + self.gamma * size_ratio
        return {
            "loss": total_loss,
            "label_loss": label_loss,
            "time_ratio": time_ratio,
            "size_ratio": size_ratio,
        }

    def _compute_ratio(
        self,
        original_flow: np.ndarray,
        adv_flow: np.ndarray,
        indices: list[int],
        flow_mask: np.ndarray | None,
    ) -> float:
        if not indices:
            return 0.0
        reference = original_flow[:, indices, :]
        candidate = adv_flow[:, indices, :]
        return _safe_l2_ratio(reference, candidate, mask=flow_mask)

    def _compute_ratio_torch(self, original_flow, adv_flow, indices: list[int], flow_mask=None):
        if not indices:
            torch_module = _import_torch()
            return torch_module.zeros((), dtype=original_flow.dtype, device=original_flow.device)
        reference = original_flow[:, indices, :]
        candidate = adv_flow[:, indices, :]
        return _safe_l2_ratio_torch(reference, candidate, mask=flow_mask)
