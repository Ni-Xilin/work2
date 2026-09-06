"""真实数据集加载模块。
这个文件负责参考第一工作点基线的数据组织方式，
把 DeepCorr300 / mDeepCorr / DeepCoFFEA 整理成第二工作点统一样本。"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from second_workpoint.config import ExperimentConfig
from second_workpoint.data.preprocessing import (
    build_deepcoffea_session,
    build_deepcorr_full_flow,
    build_sample_views,
    iter_window_starts,
)


_DEEP_CORR_SAMPLE_CACHE: dict[str, list[dict]] = {}


DEEP_CORR_RUN_NAMES = (
    "8872",
    "8802",
    "8873",
    "8803",
    "8874",
    "8804",
    "8875",
    "8876",
    "8877",
    "8878",
)


class DeepCorrRealDataset:
    """DeepCorr300 / mDeepCorr 真实数据集。

    这里直接沿用第一工作点基线的原始 pickle 样本，
    先恢复成 8 通道流量，再抽取 Tor 侧 4 通道窗口。"""

    def __init__(self, config: ExperimentConfig, split: str) -> None:
        self.config = config
        self.split = _normalize_split(config, split)
        self.dataset_name = str(config.data)
        self.data_dir = Path(config.data_path)
        self.raw_samples = self._load_raw_samples()
        self.sample_indices = self._resolve_split_indices()
        self.negative_sample_indices = (
            _build_negative_index_map(
                self.sample_indices,
                config.negative_pairs_per_sample,
                split_seed=config.split_seed,
            )
            if self.split == "test"
            else {}
        )
        self.window_starts = self._build_window_starts()

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        sample_index = int(self.sample_indices[int(index)])
        raw_sample = self.raw_samples[int(sample_index)]
        full_flow_8 = build_deepcorr_full_flow(raw_sample, self.config.flow_size)
        tor_flow = full_flow_8[np.asarray(self.config.tor_row_indices, dtype=np.int64)]

        windows = []
        for window_index, start in enumerate(self.window_starts):
            history_seq = tor_flow[:, start : start + self.config.seq_len].copy()
            future_start = start + self.config.seq_len
            future_end = future_start + self.config.pred_len
            clean_future = tor_flow[:, future_start:future_end].copy()
            windows.append(
                build_sample_views(
                    config=self.config,
                    dataset_name=self.dataset_name,
                    history_seq=history_seq,
                    clean_future=clean_future,
                    full_flow=tor_flow,
                    sample_index=int(sample_index),
                    window_index=int(window_index),
                    window_start=int(start),
                    writeback_start=int(future_start),
                    raw_length=self.config.flow_size,
                    future_valid_length=self.config.pred_len,
                )
            )

        sample = {
            "full_flow": tor_flow.astype(np.float32),
            "history_seq": np.stack([window["history_seq"] for window in windows], axis=0),
            "clean_future": np.stack([window["clean_future"] for window in windows], axis=0),
            "prompt_text": [str(window["prompt_text"]) for window in windows],
            "future_mask": np.stack([window["future_mask"] for window in windows], axis=0),
            "window_meta": np.stack([window["window_meta"] for window in windows], axis=0),
            "writeback_meta": np.stack([window["writeback_meta"] for window in windows], axis=0),
        }
        sample["target_full_flow"] = full_flow_8.astype(np.float32)
        exit_rows = [index for index in range(full_flow_8.shape[0]) if index not in self.config.tor_row_indices]
        if self.split == "test":
            mismatched_flows = []
            for negative_index in self.negative_sample_indices[int(sample_index)]:
                negative_flow = build_deepcorr_full_flow(self.raw_samples[negative_index], self.config.flow_size)
                mismatched_flow = full_flow_8.copy()
                mismatched_flow[exit_rows] = negative_flow[exit_rows]
                mismatched_flows.append(mismatched_flow.astype(np.float32))
            sample["target_negative_flow"] = np.stack(mismatched_flows, axis=0)
        sample["sample_key"] = f"{self.dataset_name}:{sample_index}"
        return sample

    def _load_raw_samples(self) -> list[dict]:
        cache_key = str(self.data_dir.resolve())
        if cache_key in _DEEP_CORR_SAMPLE_CACHE:
            return _DEEP_CORR_SAMPLE_CACHE[cache_key]
        samples: list[dict] = []
        for run_name in DEEP_CORR_RUN_NAMES:
            file_path = self.data_dir / f"{run_name}_tordata300.pickle"
            if not file_path.exists():
                raise FileNotFoundError(f"未找到数据文件: {file_path}")
            with file_path.open("rb") as file:
                samples.extend(pickle.load(file))
        if not samples:
            raise ValueError("DeepCorr 数据集为空，无法构造 dataloader。")
        _DEEP_CORR_SAMPLE_CACHE[cache_key] = samples
        return samples

    def _resolve_split_indices(self) -> np.ndarray:
        val_indices = _load_optional_pickle(self.data_dir / "val_index300.pickle")
        test_indices = _load_optional_pickle(self.data_dir / "test_index300.pickle")
        train_indices = _load_optional_pickle(self.data_dir / "train_index300.pickle")

        if self.config.use_cached_indices and val_indices is not None and test_indices is not None:
            if self.split == "train":
                if train_indices is not None:
                    indices = np.asarray(train_indices, dtype=np.int64)
                else:
                    held_out = set(int(index) for index in val_indices)
                    held_out.update(int(index) for index in test_indices)
                    remaining_indices = np.asarray(
                        [index for index in range(len(self.raw_samples)) if index not in held_out],
                        dtype=np.int64,
                    )
                    rng = np.random.RandomState(self.config.split_seed)
                    rng.shuffle(remaining_indices)
                    unused = min(self.config.deepcorr_unused_samples, max(0, remaining_indices.size - 1))
                    indices = remaining_indices[: remaining_indices.size - unused] if unused else remaining_indices
            elif self.split == "val":
                indices = np.asarray(val_indices, dtype=np.int64)
            else:
                indices = np.asarray(test_indices, dtype=np.int64)
        else:
            indices = _build_deterministic_split(
                total_size=len(self.raw_samples),
                split=self.split,
                split_seed=self.config.split_seed,
                val_size=self.config.val_samples,
                test_size=self.config.test_samples,
                unused_size=self.config.deepcorr_unused_samples,
            )

        limit = self.config.max_train_samples if self.split == "train" else self.config.max_eval_samples
        if limit > 0:
            indices = indices[:limit]
        return indices.astype(np.int64)

    def _build_window_starts(self) -> list[int]:
        starts = iter_window_starts(
            total_length=self.config.flow_size,
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
            stride=self.config.stride,
            include_tail=False,
        )
        if not starts:
            raise ValueError("DeepCorr 数据窗口参数非法，无法切出任何样本。")
        return starts


class DeepCoffeaRealDataset:
    """DeepCoFFEA 真实数据集。

    这里使用 session 级 Tor 数据构造 history/future 窗口，
    与第二工作点当前讨论的技术方案保持一致。"""

    def __init__(self, config: ExperimentConfig, split: str) -> None:
        self.config = config
        self.split = _normalize_split(config, split)
        if self.split == "val":
            raise ValueError("DeepCoFFEA 当前 dataloader 不支持 val split，请使用 eval_split=test。")
        self.dataset_name = "DeepCoFFEA"
        self.data_dir = Path(config.data_path)
        self.file_split = "train" if self.split == "train" else "test"
        self.session_data = self._load_session_data()
        self.window_specs = self._build_window_specs()

    def __len__(self) -> int:
        return int(self.window_specs.shape[0])

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        session_index, window_index, start = self.window_specs[int(index)].tolist()
        session = build_deepcoffea_session(
            self.session_data["tor_ipds"][int(session_index)],
            self.session_data["tor_sizes"][int(session_index)],
        )
        exit_session = build_deepcoffea_session(
            self.session_data["exit_ipds"][int(session_index)],
            self.session_data["exit_sizes"][int(session_index)],
        )
        raw_length = int(session.shape[1])
        history_seq = session[:, start : start + self.config.seq_len].copy()

        future_start = start + self.config.seq_len
        future_end = future_start + self.config.pred_len
        clean_future_raw = session[:, future_start:future_end].copy()
        future_valid_length = int(clean_future_raw.shape[1])
        clean_future = np.zeros((self.config.enc_in, self.config.pred_len), dtype=np.float32)
        clean_future[:, :future_valid_length] = clean_future_raw

        full_flow = np.concatenate([history_seq, clean_future], axis=1).astype(np.float32)
        sample = build_sample_views(
            config=self.config,
            dataset_name=self.dataset_name,
            history_seq=history_seq,
            clean_future=clean_future,
            full_flow=full_flow,
            sample_index=int(session_index),
            window_index=int(window_index),
            window_start=int(start),
            writeback_start=self.config.seq_len,
            raw_length=raw_length,
            future_valid_length=future_valid_length,
        )
        sample["label_text"] = str(self.session_data["labels"][int(session_index)])
        sample["target_full_flow"] = _fixed_session_window(
            session,
            start=start,
            target_length=self.config.deepcoffea_tor_len,
        )
        sample["target_exit_flow"] = _fixed_session_window(
            exit_session,
            start=start,
            target_length=self.config.deepcoffea_exit_len,
        )
        negative_exit_windows = []
        session_count = len(self.session_data["labels"])
        for offset in range(1, self.config.negative_pairs_per_sample + 1):
            negative_session_index = (int(session_index) + offset) % session_count
            negative_exit_session = build_deepcoffea_session(
                self.session_data["exit_ipds"][negative_session_index],
                self.session_data["exit_sizes"][negative_session_index],
            )
            negative_exit_windows.append(
                _fixed_session_window(
                    negative_exit_session,
                    start=start,
                    target_length=self.config.deepcoffea_exit_len,
                )
            )
        sample["target_negative_exit_flow"] = np.stack(negative_exit_windows, axis=0)
        sample["sample_key"] = f"{self.dataset_name}:{session_index}:{window_index}"
        sample["session_length"] = np.asarray(raw_length, dtype=np.int64)
        return sample

    def _load_session_data(self) -> dict[str, np.ndarray]:
        session_path = self.data_dir / f"{self.config.deepcoffea_prefix}_{self.file_split}_session.npz"
        if not session_path.exists():
            raise FileNotFoundError(f"未找到 DeepCoFFEA session 文件: {session_path}")
        payload = np.load(session_path, allow_pickle=True)
        required_keys = {"tor_ipds", "tor_sizes", "exit_ipds", "exit_sizes", "labels"}
        missing = required_keys.difference(payload.files)
        if missing:
            raise KeyError(f"DeepCoFFEA session 文件缺少字段: {sorted(missing)}")
        return {
            "tor_ipds": payload["tor_ipds"],
            "tor_sizes": payload["tor_sizes"],
            "exit_ipds": payload["exit_ipds"],
            "exit_sizes": payload["exit_sizes"],
            "labels": payload["labels"],
        }

    def _build_window_specs(self) -> np.ndarray:
        specs: list[tuple[int, int, int]] = []
        session_count = len(self.session_data["labels"])
        limit = self.config.max_train_samples if self.split == "train" else self.config.max_eval_samples
        for session_index in range(session_count):
            session = build_deepcoffea_session(
                self.session_data["tor_ipds"][session_index],
                self.session_data["tor_sizes"][session_index],
            )
            starts = iter_window_starts(
                total_length=int(session.shape[1]),
                seq_len=self.config.seq_len,
                pred_len=self.config.pred_len,
                stride=self.config.stride,
                include_tail=self.config.deepcoffea_include_tail,
            )
            for window_index, start in enumerate(starts):
                specs.append((session_index, window_index, int(start)))
                if limit > 0 and len(specs) >= limit:
                    return np.asarray(specs, dtype=np.int64)

        if not specs:
            raise ValueError("DeepCoFFEA session 切分后没有可用窗口。")

        window_specs = np.asarray(specs, dtype=np.int64)
        if limit > 0:
            window_specs = window_specs[:limit]
        return window_specs


def _fixed_session_window(session: np.ndarray, start: int, target_length: int) -> np.ndarray:
    """Extract one target-model window and zero-pad only beyond the real session tail."""

    window = np.zeros((session.shape[0], target_length), dtype=np.float32)
    source = session[:, start : start + target_length]
    window[:, : source.shape[1]] = source
    return window


def _build_negative_index_map(
    indices: np.ndarray,
    negative_count: int,
    split_seed: int = 2025,
) -> dict[int, list[int]]:
    """Reproduce Work1's seeded shuffle-and-take mismatch construction."""

    values = [int(index) for index in indices.tolist()]
    if not values:
        return {}
    if negative_count > len(values) - 1:
        raise ValueError(
            "negative_pairs_per_sample must be smaller than the number of test samples "
            "so every Work1-style mismatch is distinct."
        )
    rng = np.random.RandomState(split_seed)
    shuffled = values.copy()
    mapping: dict[int, list[int]] = {}
    for value in values:
        rng.shuffle(shuffled)
        mapping[value] = [candidate for candidate in shuffled if candidate != value][:negative_count]
    return mapping


def _normalize_split(config: ExperimentConfig, split: str) -> str:
    """把 work2 训练器使用的 split 统一到真实数据集约定。"""

    if split == "eval":
        return config.eval_split
    if split not in {"train", "val", "test"}:
        raise ValueError("split 只支持 train / val / test / eval。")
    return split


def _load_optional_pickle(path: Path) -> list[int] | None:
    """尝试读取可选的索引文件。"""

    if not path.exists():
        return None
    with path.open("rb") as file:
        payload = pickle.load(file)
    return [int(index) for index in payload]


def _build_deterministic_split(
    total_size: int,
    split: str,
    split_seed: int,
    val_size: int,
    test_size: int,
    unused_size: int = 0,
) -> np.ndarray:
    """Reproduce Work1's train/val/test/unused split with a stable seed."""

    rng = np.random.RandomState(split_seed)
    permutation = np.arange(total_size, dtype=np.int64)
    rng.shuffle(permutation)

    effective_val = min(max(0, val_size), total_size)
    remaining = max(0, total_size - effective_val)
    effective_test = min(max(0, test_size), remaining)
    remaining = max(0, remaining - effective_test)
    effective_unused = min(max(0, unused_size), remaining)
    train_end = total_size - effective_val - effective_test - effective_unused

    train_indices = permutation[:train_end]
    val_indices = permutation[train_end : train_end + effective_val]
    test_start = train_end + effective_val
    test_indices = permutation[test_start : test_start + effective_test]

    if split == "train":
        return train_indices.astype(np.int64)
    if split == "val":
        return val_indices.astype(np.int64)
    return test_indices.astype(np.int64)
