"""数据集工厂模块。
这个文件负责根据配置选择 stub 数据集或真实数据集，
让训练器不必直接依赖某个具体数据实现。"""

from __future__ import annotations

from second_workpoint.config import ExperimentConfig
from second_workpoint.data.real_dataset import DeepCoffeaRealDataset, DeepCorrRealDataset
from second_workpoint.data.stub_dataset import StubFlowDataset


def build_dataset(config: ExperimentConfig, split: str):
    """根据配置构造数据集实例。"""

    if config.data_loader == "stub":
        return StubFlowDataset(config, split=split)

    dataset_name = config.data.lower()
    if dataset_name in {"deepcorr300", "deepcorr", "mdeepcorr"}:
        return DeepCorrRealDataset(config, split=split)
    if dataset_name in {"deepcoffea", "deepcoffea_real"}:
        return DeepCoffeaRealDataset(config, split=split)

    raise ValueError(f"暂不支持的数据集类型: {config.data}")
