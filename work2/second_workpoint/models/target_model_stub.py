"""冻结目标反馈模型 stub。
这个文件用一个轻量统计分类器代替真实 DeepCorr / DeepCoFFEA 目标模型。
它的作用不是给出真实攻击指标，而是先提供一个稳定的、可复现实验接口。"""

from __future__ import annotations

import numpy as np

from second_workpoint.config import ExperimentConfig
from second_workpoint.utils.runtime import gelu


class FrozenTargetModelStub:
    def __init__(self, config: ExperimentConfig) -> None:
        rng = np.random.default_rng(config.random_seed + 151)
        input_dim = config.enc_in * 4
        hidden_dim = config.target_hidden_dim
        self.weight1 = rng.normal(0.0, 0.05, size=(input_dim, hidden_dim)).astype(np.float32)
        self.bias1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.weight2 = rng.normal(0.0, 0.05, size=(hidden_dim, 1)).astype(np.float32)
        self.bias2 = np.zeros((1,), dtype=np.float32)

    def forward(self, adv_flow: np.ndarray) -> np.ndarray:
        mean_feature = adv_flow.mean(axis=-1)
        std_feature = adv_flow.std(axis=-1)
        max_feature = adv_flow.max(axis=-1)
        min_feature = adv_flow.min(axis=-1)
        features = np.stack(
            [mean_feature, std_feature, max_feature, min_feature],
            axis=-1,
        ).reshape(adv_flow.shape[0], -1)
        hidden = gelu(features @ self.weight1 + self.bias1)
        logits = hidden @ self.weight2 + self.bias2
        return logits.astype(np.float32)

    def parameter_specs(self) -> list[tuple[str, np.ndarray, bool]]:
        return [
            ("weight1", self.weight1, False),
            ("bias1", self.bias1, False),
            ("weight2", self.weight2, False),
            ("bias2", self.bias2, False),
        ]

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value, _ in self.parameter_specs()}
