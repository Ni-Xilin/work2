"""Stub 数据集模块。
这个文件负责构造最小可跑通的流量样本。
第一版不依赖真实数据集，而是按照 Generator_Trainer 的参数习惯生成固定长度 flow，
并切出 history_seq 与 clean_future。"""

from __future__ import annotations

import math

import numpy as np

from second_workpoint.config import ExperimentConfig


class StubFlowDataset:
    def __init__(self, config: ExperimentConfig, split: str) -> None:
        if split not in {"train", "eval"}:
            raise ValueError("split 只支持 train 或 eval。")
        self.config = config
        self.split = split
        self.length = config.stub_dataset_size if split == "train" else config.eval_dataset_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(self.config.random_seed + index + self._seed_offset())
        full_flow = self._build_flow(index, rng)
        history_seq = full_flow[:, : self.config.seq_len].copy()
        clean_future = full_flow[:, self.config.seq_len : self.config.seq_len + self.config.pred_len].T.copy()
        prompt_ids = rng.integers(
            low=0,
            high=self.config.prompt_vocab_size,
            size=(self.config.prompt_len,),
            dtype=np.int64,
        )
        return {
            "full_flow": full_flow,
            "history_seq": history_seq,
            "clean_future": clean_future,
            "prompt_ids": prompt_ids,
        }

    def _seed_offset(self) -> int:
        return 0 if self.split == "train" else 100_000

    def _build_flow(self, index: int, rng: np.random.Generator) -> np.ndarray:
        time_axis = np.linspace(0.0, 1.0, self.config.flow_size, dtype=np.float32)
        full_flow = np.zeros((self.config.enc_in, self.config.flow_size), dtype=np.float32)
        phase = float(index % 7) * 0.15

        for channel in range(self.config.enc_in):
            base_wave = np.sin(2.0 * math.pi * (channel + 1) * time_axis + phase)
            trend = (channel + 1) * 0.05 * time_axis
            noise = rng.normal(0.0, 0.05, size=self.config.flow_size).astype(np.float32)
            channel_signal = base_wave + trend + noise

            if channel in self.config.time_channel_indices:
                channel_signal = np.abs(channel_signal) * 10.0 + 0.5
            elif channel in self.config.size_channel_indices:
                channel_signal = np.abs(channel_signal) * 2.0 + 0.1
            else:
                channel_signal = np.abs(channel_signal) + 0.1

            full_flow[channel] = channel_signal.astype(np.float32)

        return full_flow
