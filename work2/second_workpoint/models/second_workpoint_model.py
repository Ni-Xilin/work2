"""第二工作点主模型。
这个文件把最小版本的时序分支、视觉分支、Shared Reprogramming、冻结 backbone stub
和扰动输出头串成一个完整前向流程。当前版本只追求结构跑通，不追求真实论文效果。"""

from __future__ import annotations

import math

import numpy as np

from second_workpoint.config import ExperimentConfig
from second_workpoint.models.backbone_stub import FrozenBackboneStub
from second_workpoint.utils.runtime import gelu, layer_norm, softmax


class TemporalPatchAdapter:
    """时序分支。
    负责把历史流量切成 patch，再映射成 backbone 可消费的时序 token。"""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.patch_len = config.patch_len
        self.patch_num = config.seq_len // config.patch_len
        input_dim = config.enc_in * config.patch_len
        rng = np.random.default_rng(config.random_seed + 201)
        self.projection_weight = rng.normal(0.0, 0.05, size=(input_dim, config.d_model)).astype(np.float32)
        self.projection_bias = np.zeros((config.d_model,), dtype=np.float32)

    def forward(self, history_seq: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        batch_size = history_seq.shape[0]
        patch_view = history_seq.reshape(
            batch_size,
            self.config.enc_in,
            self.patch_num,
            self.patch_len,
        ).transpose(0, 2, 1, 3)
        flat_inputs = patch_view.reshape(batch_size, self.patch_num, -1)
        projected = flat_inputs @ self.projection_weight + self.projection_bias
        tokens = layer_norm(projected)
        return tokens.astype(np.float32), {"flat_inputs": flat_inputs}

    def apply_stub_update(
        self,
        input_mean: np.ndarray,
        token_signal: np.ndarray,
        learning_rate: float,
        update_scale: float,
    ) -> None:
        self.projection_weight -= learning_rate * update_scale * np.outer(input_mean, token_signal).astype(np.float32)
        self.projection_bias -= learning_rate * update_scale * token_signal.astype(np.float32)

    def parameter_specs(self) -> list[tuple[str, np.ndarray, bool]]:
        return [
            ("projection_weight", self.projection_weight, True),
            ("projection_bias", self.projection_bias, True),
        ]


class VisualPatchAdapter:
    """视觉分支。
    负责把历史流量转换为“伪视觉统计 patch”，再映射成视觉 token。"""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.patch_len = config.patch_len
        self.patch_num = config.seq_len // config.patch_len
        input_dim = config.enc_in * config.visual_conv_channels
        rng = np.random.default_rng(config.random_seed + 211)
        self.projection_weight = rng.normal(0.0, 0.05, size=(input_dim, config.d_model)).astype(np.float32)
        self.projection_bias = np.zeros((config.d_model,), dtype=np.float32)

    def forward(self, history_seq: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        batch_size = history_seq.shape[0]
        patch_view = history_seq.reshape(
            batch_size,
            self.config.enc_in,
            self.patch_num,
            self.patch_len,
        ).transpose(0, 2, 1, 3)
        mean_feature = patch_view.mean(axis=-1)
        std_feature = patch_view.std(axis=-1)
        max_feature = patch_view.max(axis=-1)
        min_feature = patch_view.min(axis=-1)
        stats = np.stack([mean_feature, std_feature, max_feature, min_feature], axis=-1)

        if self.config.visual_conv_channels <= stats.shape[-1]:
            expanded = stats[:, :, :, : self.config.visual_conv_channels]
        else:
            repeat_factor = math.ceil(self.config.visual_conv_channels / stats.shape[-1])
            expanded = np.tile(stats, (1, 1, 1, repeat_factor))[:, :, :, : self.config.visual_conv_channels]

        flat_inputs = expanded.reshape(batch_size, self.patch_num, -1)
        projected = flat_inputs @ self.projection_weight + self.projection_bias
        tokens = layer_norm(projected)
        return tokens.astype(np.float32), {"flat_inputs": flat_inputs}

    def apply_stub_update(
        self,
        input_mean: np.ndarray,
        token_signal: np.ndarray,
        learning_rate: float,
        update_scale: float,
    ) -> None:
        self.projection_weight -= learning_rate * update_scale * np.outer(input_mean, token_signal).astype(np.float32)
        self.projection_bias -= learning_rate * update_scale * token_signal.astype(np.float32)

    def parameter_specs(self) -> list[tuple[str, np.ndarray, bool]]:
        return [
            ("projection_weight", self.projection_weight, True),
            ("projection_bias", self.projection_bias, True),
        ]


class SharedReprogrammingLayer:
    """共享重编程层。
    负责把时序 token 和视觉 token 都映射到同一个原型语义空间。"""

    def __init__(self, num_prototypes: int, hidden_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.prototypes = rng.normal(0.0, 0.02, size=(num_prototypes, hidden_dim)).astype(np.float32)

    def forward(self, tokens: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        logits = (tokens @ self.prototypes.T) / math.sqrt(self.hidden_dim)
        weights = softmax(logits, axis=-1)
        reprogrammed = weights @ self.prototypes
        return reprogrammed.astype(np.float32), {"weights": weights}

    def apply_stub_update(
        self,
        prototype_weight_mean: np.ndarray,
        token_signal: np.ndarray,
        learning_rate: float,
        update_scale: float,
    ) -> None:
        prototype_signal = np.outer(prototype_weight_mean, token_signal)
        self.prototypes -= learning_rate * update_scale * prototype_signal.astype(np.float32)

    def parameter_specs(self) -> list[tuple[str, np.ndarray, bool]]:
        return [("prototypes", self.prototypes, True)]


class PerturbationHead:
    """扰动输出头。
    负责把 backbone 上下文压缩成未来 pred_len 区间的扰动张量。"""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        output_dim = config.pred_len * config.enc_in
        rng = np.random.default_rng(config.random_seed + 231)
        self.weight1 = rng.normal(0.0, 0.05, size=(config.d_model, config.d_model)).astype(np.float32)
        self.bias1 = np.zeros((config.d_model,), dtype=np.float32)
        self.weight2 = rng.normal(0.0, 0.05, size=(config.d_model, output_dim)).astype(np.float32)
        self.bias2 = np.zeros((output_dim,), dtype=np.float32)

    def forward(self, context_tokens: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        pooled_context = context_tokens.mean(axis=1)
        normed_context = layer_norm(pooled_context)
        hidden_pre = normed_context @ self.weight1 + self.bias1
        hidden = gelu(hidden_pre)
        perturbation = hidden @ self.weight2 + self.bias2
        perturbation = perturbation.reshape(context_tokens.shape[0], self.pred_len, self.enc_in)
        cache = {
            "pooled_context": pooled_context,
            "normed_context": normed_context,
            "hidden_pre": hidden_pre,
            "hidden": hidden,
        }
        return perturbation.astype(np.float32), cache

    def apply_stub_update(
        self,
        cache: dict[str, np.ndarray],
        grad_perturbation: np.ndarray,
        learning_rate: float,
        update_scale: float,
    ) -> np.ndarray:
        grad_output = grad_perturbation.reshape(grad_perturbation.shape[0], -1).mean(axis=0)
        hidden_mean = cache["hidden"].mean(axis=0)
        weight2_before = self.weight2.copy()
        self.weight2 -= learning_rate * update_scale * np.outer(hidden_mean, grad_output).astype(np.float32)
        self.bias2 -= learning_rate * update_scale * grad_output.astype(np.float32)

        grad_hidden = grad_output @ weight2_before.T
        gate = 0.5 + 0.5 * np.tanh(cache["hidden_pre"].mean(axis=0))
        grad_hidden = (grad_hidden * gate).astype(np.float32)
        pooled_mean = cache["normed_context"].mean(axis=0)
        self.weight1 -= learning_rate * update_scale * np.outer(pooled_mean, grad_hidden).astype(np.float32)
        self.bias1 -= learning_rate * update_scale * grad_hidden.astype(np.float32)
        return grad_hidden

    def parameter_specs(self) -> list[tuple[str, np.ndarray, bool]]:
        return [
            ("weight1", self.weight1, True),
            ("bias1", self.bias1, True),
            ("weight2", self.weight2, True),
            ("bias2", self.bias2, True),
        ]


class SecondWorkpointStubModel:
    """第二工作点总模型。
    负责串联可训练模块与冻结模块，并暴露统一的前向和 stub 更新接口。"""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.temporal_adapter = TemporalPatchAdapter(config)
        self.visual_adapter = VisualPatchAdapter(config)
        self.shared_reprogramming = SharedReprogrammingLayer(
            num_prototypes=config.reprogramming_num_prototypes,
            hidden_dim=config.d_model,
            seed=config.random_seed + 221,
        )
        self.backbone = FrozenBackboneStub(config)
        self.output_head = PerturbationHead(config)

    def forward(self, history_seq: np.ndarray, prompt_ids: np.ndarray) -> dict[str, np.ndarray | dict]:
        temporal_tokens, temporal_cache = self.temporal_adapter.forward(history_seq)
        visual_tokens, visual_cache = self.visual_adapter.forward(history_seq)
        reprogrammed_temporal, shared_temporal_cache = self.shared_reprogramming.forward(temporal_tokens)
        reprogrammed_visual, shared_visual_cache = self.shared_reprogramming.forward(visual_tokens)
        context_tokens = self.backbone.forward(prompt_ids, reprogrammed_temporal, reprogrammed_visual)
        perturbation, head_cache = self.output_head.forward(context_tokens)
        return {
            "temporal_tokens": temporal_tokens,
            "visual_tokens": visual_tokens,
            "reprogrammed_temporal": reprogrammed_temporal,
            "reprogrammed_visual": reprogrammed_visual,
            "context_tokens": context_tokens,
            "perturbation": perturbation,
            "caches": {
                "temporal": temporal_cache,
                "visual": visual_cache,
                "shared_temporal": shared_temporal_cache,
                "shared_visual": shared_visual_cache,
                "head": head_cache,
            },
        }

    def apply_stub_update(
        self,
        outputs: dict[str, np.ndarray | dict],
        grad_perturbation: np.ndarray,
        learning_rate: float,
        update_scale: float,
    ) -> None:
        caches = outputs["caches"]
        head_signal = self.output_head.apply_stub_update(
            cache=caches["head"],
            grad_perturbation=grad_perturbation,
            learning_rate=learning_rate,
            update_scale=update_scale,
        )
        temporal_input_mean = caches["temporal"]["flat_inputs"].mean(axis=(0, 1))
        visual_input_mean = caches["visual"]["flat_inputs"].mean(axis=(0, 1))
        temporal_weight_mean = outputs["caches"]["shared_temporal"]["weights"].mean(axis=(0, 1))
        visual_weight_mean = outputs["caches"]["shared_visual"]["weights"].mean(axis=(0, 1))
        prototype_weight_mean = 0.5 * (temporal_weight_mean + visual_weight_mean)

        self.temporal_adapter.apply_stub_update(
            input_mean=temporal_input_mean,
            token_signal=head_signal,
            learning_rate=learning_rate,
            update_scale=update_scale,
        )
        self.visual_adapter.apply_stub_update(
            input_mean=visual_input_mean,
            token_signal=head_signal,
            learning_rate=learning_rate,
            update_scale=update_scale,
        )
        self.shared_reprogramming.apply_stub_update(
            prototype_weight_mean=prototype_weight_mean,
            token_signal=head_signal,
            learning_rate=learning_rate,
            update_scale=update_scale,
        )

    def parameter_specs(self) -> list[tuple[str, np.ndarray, bool]]:
        specs: list[tuple[str, np.ndarray, bool]] = []
        module_specs = {
            "temporal_adapter": self.temporal_adapter.parameter_specs(),
            "visual_adapter": self.visual_adapter.parameter_specs(),
            "shared_reprogramming": self.shared_reprogramming.parameter_specs(),
            "backbone": self.backbone.parameter_specs(),
            "output_head": self.output_head.parameter_specs(),
        }
        for module_name, entries in module_specs.items():
            for parameter_name, value, trainable in entries:
                specs.append((f"{module_name}.{parameter_name}", value, trainable))
        return specs

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value, _ in self.parameter_specs()}
