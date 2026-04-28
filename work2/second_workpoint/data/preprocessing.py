"""数据预处理辅助模块。

这个文件负责把真实数据集中的原始流量或会话整理成统一样本接口。
当前只保留真实训练主线，因此 prompt 侧输出的是自然语言 `prompt_text`，
不再生成任何哈希 `prompt_ids` 占位输入。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from second_workpoint.config import ExperimentConfig


DEEP_CORR_TOR_ROWS = (0, 3, 4, 7)
PROMPT_FEATURE_ORDER = (
    "nonzero_rate",
    "time_mean",
    "time_cv",
    "size_mean_abs",
    "size_cv",
    "burst_ratio",
    "trend_ratio",
    "dir_balance",
)


def pad_or_truncate_1d(values: np.ndarray | list[float], target_length: int) -> np.ndarray:
    """把一维序列裁剪或补零到固定长度。"""

    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape[0] >= target_length:
        return array[:target_length].astype(np.float32)

    padded = np.zeros((target_length,), dtype=np.float32)
    padded[: array.shape[0]] = array
    return padded


def build_deepcorr_full_flow(sample_dict: dict, flow_size: int) -> np.ndarray:
    """把 DeepCorr 原始字典样本整理成 8 通道固定长度张量。"""

    full_flow = np.zeros((8, flow_size), dtype=np.float32)
    full_flow[0] = pad_or_truncate_1d(sample_dict["here"][0]["<-"], flow_size) * 1000.0
    full_flow[1] = pad_or_truncate_1d(sample_dict["there"][0]["->"], flow_size) * 1000.0
    full_flow[2] = pad_or_truncate_1d(sample_dict["there"][0]["<-"], flow_size) * 1000.0
    full_flow[3] = pad_or_truncate_1d(sample_dict["here"][0]["->"], flow_size) * 1000.0
    full_flow[4] = pad_or_truncate_1d(sample_dict["here"][1]["<-"], flow_size) / 1000.0
    full_flow[5] = pad_or_truncate_1d(sample_dict["there"][1]["->"], flow_size) / 1000.0
    full_flow[6] = pad_or_truncate_1d(sample_dict["there"][1]["<-"], flow_size) / 1000.0
    full_flow[7] = pad_or_truncate_1d(sample_dict["here"][1]["->"], flow_size) / 1000.0
    return full_flow


def build_deepcoffea_session(tor_ipds: np.ndarray, tor_sizes: np.ndarray) -> np.ndarray:
    """把 DeepCoFFEA 的 Tor 侧 session 序列整理成 2 通道张量。"""

    ipd = np.asarray(tor_ipds, dtype=np.float32).reshape(-1)
    size = np.asarray(tor_sizes, dtype=np.float32).reshape(-1)
    valid_length = min(ipd.shape[0], size.shape[0])
    if valid_length <= 0:
        return np.zeros((2, 0), dtype=np.float32)
    return np.stack([ipd[:valid_length], size[:valid_length]], axis=0).astype(np.float32)


def iter_window_starts(
    total_length: int,
    seq_len: int,
    pred_len: int,
    stride: int,
    include_tail: bool,
) -> list[int]:
    """根据现有工程习惯产生滑窗起点。

    - 对固定长度 flow，通常只保留 future 完整的窗口。
    - 对 session 级数据，可以额外保留一个尾部窗口，让 future 不足部分补零。"""

    if total_length < seq_len:
        return []

    remaining = total_length - seq_len - pred_len
    if not include_tail:
        if remaining < 0:
            return []
        max_full_index = remaining // stride
        return [window_index * stride for window_index in range(max_full_index + 1)]

    if remaining < 0:
        return [0]

    # 这里故意沿用 Generator_Trainer 中 split_time_series 的尾窗处理方式：
    # 在最后一个完整 future 窗口之后，再补一个“history 完整、future 可能不足”的窗口。
    max_tail_index = remaining // stride + 1
    return [window_index * stride for window_index in range(max_tail_index + 1)]


def build_temporal_view(history_seq: np.ndarray, patch_len: int) -> np.ndarray:
    """把历史窗口整理成 patch 级时序视图。"""

    channels, seq_len = history_seq.shape
    if seq_len % patch_len != 0:
        raise ValueError("history_seq 的长度必须能被 patch_len 整除。")

    patch_num = seq_len // patch_len
    patch_view = history_seq.reshape(channels, patch_num, patch_len).transpose(1, 0, 2)
    return patch_view.reshape(patch_num, channels * patch_len).astype(np.float32)


def build_visual_view(history_seq: np.ndarray, patch_len: int) -> np.ndarray:
    """把历史窗口整理成轻量视觉化结构视图。

    每个 patch、每个通道抽取 6 个统计量：
    mean / std / max / min / grad_energy / density_mean。"""

    channels, seq_len = history_seq.shape
    if seq_len % patch_len != 0:
        raise ValueError("history_seq 的长度必须能被 patch_len 整除。")

    patch_num = seq_len // patch_len
    patch_view = history_seq.reshape(channels, patch_num, patch_len).transpose(1, 0, 2)

    mean_feature = patch_view.mean(axis=-1)
    std_feature = patch_view.std(axis=-1)
    max_feature = patch_view.max(axis=-1)
    min_feature = patch_view.min(axis=-1)

    if patch_len > 1:
        gradients = np.diff(patch_view, axis=-1)
        grad_energy = np.mean(np.abs(gradients), axis=-1)
    else:
        grad_energy = np.zeros((patch_num, channels), dtype=np.float32)

    density_mean = np.mean(np.abs(patch_view) > 1e-6, axis=-1).astype(np.float32)

    descriptor = np.stack(
        [
            mean_feature,
            std_feature,
            max_feature,
            min_feature,
            grad_energy,
            density_mean,
        ],
        axis=-1,
    )
    return descriptor.reshape(patch_num, channels * 6).astype(np.float32)


def compute_prompt_features(
    history_seq: np.ndarray,
    time_channel_indices: list[int],
    size_channel_indices: list[int],
) -> dict[str, float]:
    """从历史窗口提取 prompt 摘要统计量。"""

    absolute_history = np.abs(history_seq.astype(np.float32))
    nonzero_rate = float(np.mean(absolute_history > 1e-6))

    time_values = absolute_history[time_channel_indices].reshape(-1) if time_channel_indices else np.zeros((1,), dtype=np.float32)
    size_values = absolute_history[size_channel_indices].reshape(-1) if size_channel_indices else np.zeros((1,), dtype=np.float32)
    flattened = absolute_history.reshape(-1)

    time_mean = float(np.mean(time_values))
    time_cv = _safe_cv(time_values)
    size_mean_abs = float(np.mean(size_values))
    size_cv = _safe_cv(size_values)
    burst_ratio = float(np.max(flattened) / (np.mean(flattened) + 1e-6))

    front = absolute_history[:, : max(1, history_seq.shape[1] // 4)]
    tail = absolute_history[:, -max(1, history_seq.shape[1] // 4) :]
    trend_ratio = float((np.mean(tail) - np.mean(front)) / (np.mean(flattened) + 1e-6))

    positive_mass = float(np.sum(np.abs(history_seq[history_seq > 0])))
    negative_mass = float(np.sum(np.abs(history_seq[history_seq < 0])))
    dir_balance = float((positive_mass - negative_mass) / (positive_mass + negative_mass + 1e-6))

    return {
        "nonzero_rate": nonzero_rate,
        "time_mean": time_mean,
        "time_cv": time_cv,
        "size_mean_abs": size_mean_abs,
        "size_cv": size_cv,
        "burst_ratio": burst_ratio,
        "trend_ratio": trend_ratio,
        "dir_balance": dir_balance,
    }


def build_prompt_text(
    dataset_name: str,
    prompt_features: dict[str, float],
    seq_len: int,
    pred_len: int,
    patch_len: int,
) -> str:
    """构造固定字段 prompt 文本。"""

    return (
        f"dataset={dataset_name} seq_len={seq_len} pred_len={pred_len} patch_len={patch_len} "
        f"nonzero_rate={prompt_features['nonzero_rate']:.4f} "
        f"time_mean={prompt_features['time_mean']:.4f} "
        f"time_cv={prompt_features['time_cv']:.4f} "
        f"size_mean_abs={prompt_features['size_mean_abs']:.4f} "
        f"size_cv={prompt_features['size_cv']:.4f} "
        f"burst_ratio={prompt_features['burst_ratio']:.4f} "
        f"trend_ratio={prompt_features['trend_ratio']:.4f} "
        f"dir_balance={prompt_features['dir_balance']:.4f}"
    )


def build_sample_views(
    config: "ExperimentConfig",
    dataset_name: str,
    history_seq: np.ndarray,
    clean_future: np.ndarray,
    full_flow: np.ndarray,
    sample_index: int,
    window_index: int,
    window_start: int,
    writeback_start: int,
    raw_length: int,
    future_valid_length: int,
) -> dict[str, np.ndarray]:
    """把单个窗口整理成第二工作点统一样本接口。"""

    prompt_features = compute_prompt_features(
        history_seq=history_seq,
        time_channel_indices=config.time_channel_indices,
        size_channel_indices=config.size_channel_indices,
    )
    prompt_text = build_prompt_text(
        dataset_name=dataset_name,
        prompt_features=prompt_features,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        patch_len=config.patch_len,
    )
    prompt_feature_vector = np.asarray(
        [prompt_features[name] for name in PROMPT_FEATURE_ORDER],
        dtype=np.float32,
    )
    future_mask = np.zeros((config.pred_len,), dtype=np.float32)
    future_mask[:future_valid_length] = 1.0

    return {
        "full_flow": full_flow.astype(np.float32),
        "history_seq": history_seq.astype(np.float32),
        "clean_future": clean_future.T.astype(np.float32),
        "x_ts": build_temporal_view(history_seq, config.patch_len),
        "x_vis": build_visual_view(history_seq, config.patch_len),
        "prompt_text": prompt_text,
        "prompt_features": prompt_feature_vector,
        "future_mask": future_mask,
        "window_meta": np.asarray(
            [sample_index, window_index, window_start, raw_length, future_valid_length],
            dtype=np.int64,
        ),
        "writeback_meta": np.asarray([writeback_start, future_valid_length], dtype=np.int64),
    }


def _safe_cv(values: np.ndarray) -> float:
    """计算稳健版本的变异系数。"""

    array = np.asarray(values, dtype=np.float32).reshape(-1)
    mean_value = float(np.mean(np.abs(array)))
    return float(np.std(array) / (mean_value + 1e-6))
