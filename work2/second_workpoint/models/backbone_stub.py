"""冻结基模 stub。
这个文件用一个轻量的冻结上下文编码器代替真实 LLM backbone。
它保留“冻结主干、只让梯度语义穿过输入 token”的结构位置，便于后续替换成真实基模。"""

from __future__ import annotations

import numpy as np

from second_workpoint.config import ExperimentConfig


class FrozenBackboneStub:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.max_context_tokens = config.prompt_len + 2 * (config.seq_len // config.patch_len)
        rng = np.random.default_rng(config.random_seed + 101)
        self.prompt_embedding = rng.normal(
            loc=0.0,
            scale=0.02,
            size=(config.prompt_vocab_size, config.d_model),
        ).astype(np.float32)
        self.position_embedding = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(1, self.max_context_tokens, config.d_model),
        ).astype(np.float32)
        self.mix_weights = rng.normal(
            loc=0.0,
            scale=0.03,
            size=(max(1, config.llm_layers), config.d_model, config.d_model),
        ).astype(np.float32)
        self.mix_bias = np.zeros((max(1, config.llm_layers), config.d_model), dtype=np.float32)

    def forward(
        self,
        prompt_ids: np.ndarray,
        time_tokens: np.ndarray,
        visual_tokens: np.ndarray,
    ) -> np.ndarray:
        prompt_embeddings = self.prompt_embedding[prompt_ids]
        context = np.concatenate([prompt_embeddings, time_tokens, visual_tokens], axis=1)
        context = context + self.position_embedding[:, : context.shape[1], :]

        for layer_index in range(self.mix_weights.shape[0]):
            mixed = np.tensordot(context, self.mix_weights[layer_index], axes=([2], [0]))
            mixed = mixed + self.mix_bias[layer_index]
            context = 0.65 * context + 0.35 * np.tanh(mixed)

        return context.astype(np.float32)

    def parameter_specs(self) -> list[tuple[str, np.ndarray, bool]]:
        return [
            ("prompt_embedding", self.prompt_embedding, False),
            ("position_embedding", self.position_embedding, False),
            ("mix_weights", self.mix_weights, False),
            ("mix_bias", self.mix_bias, False),
        ]

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value, _ in self.parameter_specs()}
