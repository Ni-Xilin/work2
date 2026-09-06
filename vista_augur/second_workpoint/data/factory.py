"""数据集工厂模块。

第二工作点当前只支持真实数据路径，
这里负责在 DeepCorr 和 DeepCoFFEA 两个真实数据集实现之间切换。
"""

from __future__ import annotations

from second_workpoint.config import ExperimentConfig
from second_workpoint.data.real_dataset import DeepCoffeaRealDataset, DeepCorrRealDataset, MDeepCorrRealDataset


def build_dataset(config: ExperimentConfig, split: str):
    """根据配置构造数据集实例。"""

    dataset_name = config.data.lower()
    if dataset_name in {"deepcorr300", "deepcorr"}:
        return DeepCorrRealDataset(config, split=split)
    if dataset_name == "mdeepcorr":
        return MDeepCorrRealDataset(config, split=split)
    if dataset_name in {"deepcoffea", "deepcoffea_real"}:
        return DeepCoffeaRealDataset(config, split=split)

    raise ValueError(f"暂不支持的数据集类型: {config.data}")
