"""第二工作点真实训练配置。

当前仓库只保留 ``torch_real`` 主线：
- 真实数据加载
- 冻结 Hugging Face backbone
- 冻结 torch target model
- 可切换的 ``learned_prototypes`` / ``text_prototypes`` 语义对齐
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    # 随机种子，控制数据切分、参数初始化和采样顺序的可复现性。
    random_seed: int = 2025
    # 是否进入训练模式；1 表示训练，0 表示只做评估。
    is_training: int = 1
    # 运行名标识，会参与最终 checkpoint 目录命名。
    model_id: str = "second_workpoint_torch_real"
    # 主模型类名；当前真实主线固定为 SecondWorkpointTorchRealModel。
    model: str = "SecondWorkpointTorchRealModel"
    # 目标模型名称，决定使用哪一种 target feedback 网络。
    target_model: str = "Deepcorr300"
    # 目标模型加载模式；当前只支持真实 torch target。
    target_model_mode: str = "torch"
    # target model 代码根目录，用于拼接默认权重路径和导入路径。
    target_model_root: str = "target_model"
    # 扰动类型标识，沿用旧工程命名，当前主线默认同时控制时间和大小。
    adv_type: str = "time_and_size"
    # 目标模型权重文件路径；为空时走适配器内部默认路径。
    target_model_path: str = ""
    # target model 前向时使用的 dropout，通常保持为 0.0 以稳定反馈。
    target_model_dropout: float = 0.0
    # DeepCoFFEA 中 tor 侧期望的输入序列长度。
    deepcoffea_tor_len: int = 500
    # DeepCoFFEA 中 exit 侧期望的输入序列长度。
    deepcoffea_exit_len: int = 800
    # DeepCoFFEA 缺失 exit 输入时的补齐策略；当前只支持补零。
    deepcoffea_missing_exit_policy: str = "zero"
    # backbone 加载模式；当前只支持冻结的 Hugging Face backbone。
    backbone_mode: str = "hf_frozen"
    # backbone 的 Hugging Face 仓库名，作为本地路径不可用时的后备来源。
    backbone_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    # backbone 的本地模型目录，推荐显式指向已下载好的基模路径。
    backbone_model_path: str = "base_models/Qwen2.5-3B-Instruct"
    # backbone 权重加载 dtype，可控制显存和数值精度折中。
    backbone_dtype: str = "float16"
    # backbone 量化模式；none 表示不量化，4bit/8bit 用于低显存 bring-up。
    backbone_quantization: str = "none"
    # 冻结 backbone 放置的设备。
    backbone_device: str = "cuda:0"
    # 冻结 backbone 的第二设备；为空表示 backbone 不做分层切卡。
    backbone_secondary_device: str = ""
    # backbone 分层切卡的起始层索引；0 表示禁用双卡分层。
    backbone_split_layer_index: int = 0
    # 可训练 generator 模块放置的设备。
    trainable_device: str = "cuda:1"
    # 冻结 target model 放置的设备，通常与 trainable_device 保持一致。
    target_device: str = "cuda:1"
    # 单条完整流量的总长度，必须覆盖历史窗口和未来窗口。
    flow_size: int = 300
    # 数据集名称，决定选择 DeepCorr 还是 DeepCoFFEA 数据接口。
    data: str = "Deepcorr300"
    # 数据加载模式；当前只支持真实数据集。
    data_loader: str = "real"
    # checkpoint 与训练摘要的输出根目录。
    checkpoints: str = "vista_augur/outputs/checkpoints"
    # 数据集根路径，不同任务需指向不同的数据目录。
    data_path: str = "datasets/deepcorr"
    # 历史输入窗口长度，作为 generator 的输入上下文。
    seq_len: int = 96
    # 未来预测/扰动窗口长度，也就是本轮要写回的 future 长度。
    pred_len: int = 48
    # patch 切分长度，要求 seq_len 能被它整除。
    patch_len: int = 4
    # 滑窗步长，决定样本构造时窗口向前移动的间隔。
    stride: int = 48
    # patch 对齐策略；当前保留为兼容字段。
    padding_patch: str = "end"
    # 是否使用按变量独立的头部逻辑；当前主要保留为兼容字段。
    individual: int = 1
    # 历史工程中的目标维度标记；当前保留为兼容字段。
    Dtarget: int = 1
    # 历史工程中的网络深度超参数；当前保留为兼容字段。
    depth: int = 3
    # 历史工程中的尺度放大因子；当前保留为兼容字段。
    scale_factor: int = 2
    # 历史工程中的层数超参数；当前保留为兼容字段。
    n_layers: int = 2
    # 历史工程中的 attention dropout；当前不是 backbone 的直接控制旋钮。
    att_dropout: float = 0.1
    # 输出头或历史结构中的 dropout 配置；当前主要保留为兼容字段。
    head_dropout: float = 0.0
    # generator 输入通道数；DeepCorr 常见为 4，DeepCoFFEA 常见为 2。
    enc_in: int = 4
    # 训练侧的基础工作宽度/隐藏维度，通常与 trainable_width 对齐。
    d_model: int = 128
    # reprogramming multi-head attention 的头数。
    n_heads: int = 4
    # 历史工程中的前馈层宽度；当前保留为兼容字段。
    d_ff: int = 256
    # 历史工程中的缩放因子；当前保留为兼容字段。
    factor: int = 1
    # 历史工程中的激活函数名；当前保留为兼容字段。
    activation: str = "gelu"
    # dataloader worker 数；本项目通常在 debug 阶段保持为 0。
    num_workers: int = 0
    # 历史工程中的实验重复次数标记；当前 trainer 不直接依赖它。
    itr: int = 1
    # 总训练轮数。
    train_epochs: int = 2
    # 物理 batch size，即一次真正送入显存的样本条数。
    batch_size: int = 4
    # 早停 patience；当前没有早停主逻辑，保留为兼容字段。
    patience: int = 100
    # Adam 学习率。
    learning_rate: float = 1e-3
    # 学习率调度策略名；当前主线默认 constant。
    lradj: str = "constant"
    # OneCycle 等调度策略中的 warmup 比例；当前保留为兼容字段。
    pct_start: float = 0.3
    # 是否优先使用 GPU。
    use_gpu: bool = True
    # 旧式单卡编号后备字段；当前更推荐显式写 backbone/trainable/target_device。
    gpu: int = 0
    # 是否启用多 GPU 旧式开关；当前真实主线不依赖它做自动并行。
    use_multi_gpu: bool = False
    # 旧工程风格的设备编号字符串；当前主要作为兼容字段保留。
    devices: str = "0"
    # 数据切分随机种子，用于无缓存索引时稳定划分 train/val/test。
    split_seed: int = 2025
    # 评估使用 val 还是 test；DeepCoFFEA 当前只能用 test。
    eval_split: str = "val"
    # 无缓存切分时验证集样本数。
    val_samples: int = 1000
    # 无缓存切分时测试集样本数。
    test_samples: int = 1000
    # 是否优先读取已有的 train/val/test 索引缓存。
    use_cached_indices: bool = True
    # 训练样本上限；0 表示不限制，smoke 阶段可用它快速截断。
    max_train_samples: int = 0
    # 评估样本上限；0 表示不限制。
    max_eval_samples: int = 0
    # prompt tokenizer 的最大截断长度。
    backbone_prompt_max_tokens: int = 64
    # generator 稳定语义空间宽度，也是 reprogramming 的主工作维度。
    trainable_width: int = 128
    # visual 分支卷积/统计展开后的通道数。
    visual_conv_channels: int = 8
    # 语义原型数量，决定重编程层可用的 prototype 容量。
    reprogramming_num_prototypes: int = 32
    # 语义对齐模式：learned_prototypes 或 text_prototypes。
    semantic_alignment_mode: str = "text_prototypes"
    # 历史工程中的 LLM 层数占位字段；当前不直接决定真实 backbone 层数。
    llm_layers: int = 2
    # DeepCorr 中 tor 视角 4 行写回到完整 8 行流量时的目标行索引。
    tor_row_indices: list[int] = field(default_factory=lambda: [0, 3, 4, 7])
    # DeepCoFFEA session 文件名前缀，用于定位数据文件集合。
    deepcoffea_prefix: str = "d3_ws5_nw11_thr20_tl500_el800_nt1000"
    # DeepCoFFEA 切窗时是否保留尾部不完整的 future 窗口。
    deepcoffea_include_tail: bool = True
    # 参与时间开销统计和 prompt 摘要构造的通道索引。
    time_channel_indices: list[int] = field(default_factory=lambda: [0, 1])
    # 参与大小开销统计和 prompt 摘要构造的通道索引。
    size_channel_indices: list[int] = field(default_factory=lambda: [2, 3])
    # 目标误导损失项的权重。
    beta: float = 1.0
    # 时间开销约束项的权重。
    alpha: float = 1.5
    # 大小开销约束项的权重。
    gamma: float = 0.9
    # 梯度裁剪阈值。
    max_grad_norm: float = 1.0
    # 梯度累积步数；有效 batch size 约等于 batch_size * gradient_accumulation_steps。
    gradient_accumulation_steps: int = 1
    # 训练阶段最大 step 数；0 表示按完整 epoch 运行。
    max_train_steps: int = 0
    # 评估阶段最大 step 数；0 表示跑完整 eval dataloader。
    max_eval_steps: int = 0
    # 每隔多少个训练 step 打印一次日志。
    log_interval: int = 1
    # 训练后端标识；当前真实主线固定为 torch_real。
    backend: str = "torch_real"

    def validate(self) -> None:
        if self.flow_size < self.seq_len + self.pred_len:
            raise ValueError("flow_size 必须不小于 seq_len + pred_len。")
        if self.seq_len % self.patch_len != 0:
            raise ValueError("seq_len 必须能被 patch_len 整除。")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除。")
        if self.enc_in <= 0:
            raise ValueError("enc_in 必须大于 0。")
        if self.train_epochs <= 0:
            raise ValueError("train_epochs 必须大于 0。")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0。")
        if self.backbone_prompt_max_tokens < 0:
            raise ValueError("backbone_prompt_max_tokens 不能为负数。")
        if self.trainable_width <= 0:
            raise ValueError("trainable_width 必须大于 0。")
        if self.trainable_width % self.n_heads != 0:
            raise ValueError("trainable_width 必须能被 n_heads 整除。")
        if self.data_loader != "real":
            raise ValueError("当前第二工作点只支持 data_loader=real。")
        if self.target_model_mode != "torch":
            raise ValueError("当前第二工作点只支持 target_model_mode=torch。")
        if self.backbone_mode != "hf_frozen":
            raise ValueError("当前第二工作点只支持 backbone_mode=hf_frozen。")
        if self.backbone_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("backbone_dtype must be float32, float16, or bfloat16.")
        if self.backbone_quantization not in {"none", "4bit", "8bit"}:
            raise ValueError("backbone_quantization must be none, 4bit, or 8bit.")
        if self.backbone_split_layer_index < 0:
            raise ValueError("backbone_split_layer_index must be >= 0.")
        if self.target_model_dropout < 0.0 or self.target_model_dropout >= 1.0:
            raise ValueError("target_model_dropout must be in [0, 1).")
        if self.deepcoffea_tor_len <= 0 or self.deepcoffea_exit_len <= 0:
            raise ValueError("deepcoffea_tor_len and deepcoffea_exit_len must be positive.")
        if self.deepcoffea_missing_exit_policy not in {"zero"}:
            raise ValueError("deepcoffea_missing_exit_policy currently supports only zero.")
        if self.eval_split not in {"val", "test"}:
            raise ValueError("eval_split 只支持 val 或 test。")
        if self.val_samples < 0 or self.test_samples < 0:
            raise ValueError("val_samples 和 test_samples 不能为负数。")
        if self.max_train_samples < 0 or self.max_eval_samples < 0:
            raise ValueError("max_train_samples 和 max_eval_samples 不能为负数。")
        if self.max_train_steps < 0 or self.max_eval_steps < 0:
            raise ValueError("max_train_steps 和 max_eval_steps 不能为负数。")
        if self.visual_conv_channels <= 0:
            raise ValueError("visual_conv_channels 必须大于 0。")
        if self.reprogramming_num_prototypes <= 0:
            raise ValueError("reprogramming_num_prototypes 必须大于 0。")
        if self.semantic_alignment_mode not in {"learned_prototypes", "text_prototypes"}:
            raise ValueError("semantic_alignment_mode 只支持 learned_prototypes 或 text_prototypes。")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps 必须大于 0。")
        if self.backend != "torch_real":
            raise ValueError("当前第二工作点只支持 backend=torch_real。")
        if self.data_loader == "real" and self.data.lower() in {"deepcoffea", "deepcoffea_real"} and self.eval_split == "val":
            raise ValueError("DeepCoFFEA 当前只有 train/test session 文件，eval_split 不能设置为 val。")
        if not self.backbone_model_name and not self.backbone_model_path:
            raise ValueError("torch_real backend 需要 backbone_model_name 或 backbone_model_path。")
        if self.model == "SecondWorkpointTorch":
            self.model = "SecondWorkpointTorchRealModel"
        if self.backbone_model_path and not Path(self.backbone_model_path).exists():
            raise ValueError(f"backbone_model_path 不存在: {self.backbone_model_path}")
        if self.target_model_path and not Path(self.target_model_path).exists():
            raise ValueError(f"target_model_path 不存在: {self.target_model_path}")
        self._validate_source_indices(self.tor_row_indices, "tor_row_indices", upper_bound=8)
        self._validate_channel_indices(self.time_channel_indices, "time_channel_indices")
        self._validate_channel_indices(self.size_channel_indices, "size_channel_indices")

    def _validate_channel_indices(self, indices: list[int], field_name: str) -> None:
        for index in indices:
            if index < 0 or index >= self.enc_in:
                raise ValueError(f"{field_name} 中存在越界索引 {index}。")

    def _validate_source_indices(self, indices: list[int], field_name: str, upper_bound: int) -> None:
        for index in indices:
            if index < 0 or index >= upper_bound:
                raise ValueError(f"{field_name} 中存在越界索引 {index}。")

    def setting_name(self) -> str:
        return (
            f"{self.model_id}_{self.model}_{self.data}"
            f"_sl{self.seq_len}_pl{self.pred_len}_dm{self.d_model}_nh{self.n_heads}"
            f"_sa{self.semantic_alignment_mode}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    config = ExperimentConfig(**payload)
    config.validate()
    return config
