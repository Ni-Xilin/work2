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
    random_seed: int = 2025
    is_training: int = 1
    model_id: str = "second_workpoint_torch_real"
    model: str = "SecondWorkpointTorchRealModel"
    target_model: str = "Deepcorr300"
    target_model_mode: str = "torch"
    target_model_root: str = "target_model"
    adv_type: str = "time_and_size"
    target_model_path: str = ""
    target_model_dropout: float = 0.0
    deepcoffea_tor_len: int = 500
    deepcoffea_exit_len: int = 800
    deepcoffea_missing_exit_policy: str = "zero"
    backbone_mode: str = "hf_frozen"
    backbone_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    backbone_model_path: str = "base_models/Qwen2.5-3B-Instruct"
    backbone_dtype: str = "float16"
    backbone_quantization: str = "none"
    backbone_device: str = "cuda:0"
    trainable_device: str = "cuda:1"
    target_device: str = "cuda:1"
    flow_size: int = 300
    data: str = "Deepcorr300"
    data_loader: str = "real"
    checkpoints: str = "work2/outputs/checkpoints"
    data_path: str = "Generator_Trainer/target_model/deepcorr/dataset"
    seq_len: int = 96
    pred_len: int = 48
    patch_len: int = 4
    stride: int = 48
    padding_patch: str = "end"
    individual: int = 1
    Dtarget: int = 1
    depth: int = 3
    scale_factor: int = 2
    n_layers: int = 2
    att_dropout: float = 0.1
    head_dropout: float = 0.0
    enc_in: int = 4
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    factor: int = 1
    activation: str = "gelu"
    num_workers: int = 0
    itr: int = 1
    train_epochs: int = 2
    batch_size: int = 4
    patience: int = 100
    learning_rate: float = 1e-3
    lradj: str = "constant"
    pct_start: float = 0.3
    use_gpu: bool = True
    gpu: int = 0
    use_multi_gpu: bool = False
    devices: str = "0"
    split_seed: int = 2025
    eval_split: str = "val"
    val_samples: int = 1000
    test_samples: int = 1000
    use_cached_indices: bool = True
    max_train_samples: int = 0
    max_eval_samples: int = 0
    backbone_prompt_max_tokens: int = 64
    trainable_width: int = 128
    visual_conv_channels: int = 8
    reprogramming_num_prototypes: int = 32
    semantic_alignment_mode: str = "text_prototypes"
    llm_layers: int = 2
    tor_row_indices: list[int] = field(default_factory=lambda: [0, 3, 4, 7])
    deepcoffea_prefix: str = "d3_ws5_nw11_thr20_tl500_el800_nt1000"
    deepcoffea_include_tail: bool = True
    time_channel_indices: list[int] = field(default_factory=lambda: [0, 1])
    size_channel_indices: list[int] = field(default_factory=lambda: [2, 3])
    beta: float = 1.0
    alpha: float = 1.5
    gamma: float = 0.9
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    max_train_steps: int = 0
    max_eval_steps: int = 0
    log_interval: int = 1
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
