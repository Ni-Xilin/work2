"""Target model factory for the second work point.

当前只保留真实 torch target model 路径：
根据配置加载最外层 `target_model/` 中的真实代码与权重，
并在训练时保持其参数冻结。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from second_workpoint.config import ExperimentConfig


def build_target_model(config: ExperimentConfig):
    return build_torch_target_model(config)


def build_torch_target_model(config: ExperimentConfig):
    target_name = config.target_model.lower()
    if target_name in {"deepcorr300", "deepcorr300torch"}:
        return DeepCorrTorchTarget(config, variant="300")
    if target_name in {"deepcorr100", "deepcorr100torch"}:
        return DeepCorrTorchTarget(config, variant="100")
    if target_name in {"deepcorr700"}:
        return DeepCorrTorchTarget(config, variant="700")
    if target_name in {"mdeepcorr", "mdeepcorrtorch"}:
        return MDeepCorrTorchTarget(config)
    if target_name in {"deepcoffea", "deepcoffea_real", "deepcoffeatorch"}:
        return DeepCoFFEATorchTarget(config)
    raise ValueError(f"Unsupported torch target_model: {config.target_model}")


class DeepCorrTorchTarget:
    _MODEL_FILES = {
        "100": "Deepcorr100.py",
        "300": "Deepcorr300.py",
        "700": "Deepcorr700.py",
    }
    _DEFAULT_CHECKPOINTS = {
        "100": "deepcorr/deepcorr100/tor_199_epoch10_acc0.66.pth",
        "300": "deepcorr/deepcorr300/tor_199_epoch23_acc0.82dict.pth",
        "700": "deepcorr/deepcorr700/tor700_199_epoch11_acc0.88.pth",
    }
    _EXPECTED_LENGTHS = {
        "100": 100,
        "300": 300,
        "700": 700,
    }

    def __init__(self, config: ExperimentConfig, variant: str) -> None:
        self.config = config
        self.variant = variant
        self.torch = _import_torch()
        self.device = _torch_device(self.torch, config)
        root = Path(config.target_model_root)
        module = _load_module(root / self._MODEL_FILES[variant], f"work2_target_deepcorr_{variant}")
        self.model = module.Model().float().to(self.device)
        checkpoint = _resolve_checkpoint(config, self._DEFAULT_CHECKPOINTS[variant])
        state_dict = _torch_load(self.torch, checkpoint, self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, adv_flow, exit_flow=None):
        flow = _prepare_deepcorr_flow_torch(
            adv_flow=adv_flow,
            expected_length=self._EXPECTED_LENGTHS[self.variant],
            tor_row_indices=self.config.tor_row_indices,
            torch_module=self.torch,
            device=self.device,
        )
        return self.model(flow, dropout=float(self.config.target_model_dropout))

    def parameters(self):
        return self.model.parameters()


class MDeepCorrTorchTarget:
    """Work1 two-stage m-DeepCorr cascade: DeepCorr100 then DeepCorr700."""

    _DEFAULT_CHECKPOINT_100 = "deepcorr/deepcorr100/tor_199_epoch10_acc0.66.pth"
    _DEFAULT_CHECKPOINT_700 = "deepcorr/deepcorr700/tor700_199_epoch11_acc0.88.pth"

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.torch = _import_torch()
        self.device = _torch_device(self.torch, config)
        root = Path(config.target_model_root)

        module100 = _load_module(root / "Deepcorr100.py", "work2_target_mdeepcorr_100")
        module700 = _load_module(root / "Deepcorr700.py", "work2_target_mdeepcorr_700")
        self.model100 = module100.Model().float().to(self.device)
        self.model700 = module700.Model().float().to(self.device)

        checkpoint100 = _resolve_explicit_checkpoint(
            config.mdeepcorr100_model_path,
            root / self._DEFAULT_CHECKPOINT_100,
            "mdeepcorr100_model_path",
        )
        checkpoint700 = _resolve_explicit_checkpoint(
            config.mdeepcorr700_model_path,
            root / self._DEFAULT_CHECKPOINT_700,
            "mdeepcorr700_model_path",
        )
        self.model100.load_state_dict(_torch_load(self.torch, checkpoint100, self.device))
        self.model700.load_state_dict(_torch_load(self.torch, checkpoint700, self.device))
        self.model100.eval()
        self.model700.eval()
        for parameter in list(self.model100.parameters()) + list(self.model700.parameters()):
            parameter.requires_grad = False

    def forward(self, adv_flow, exit_flow=None):
        flow100 = _prepare_deepcorr_flow_torch(
            adv_flow=adv_flow,
            expected_length=100,
            tor_row_indices=self.config.tor_row_indices,
            torch_module=self.torch,
            device=self.device,
        )
        flow700 = _prepare_deepcorr_flow_torch(
            adv_flow=adv_flow,
            expected_length=700,
            tor_row_indices=self.config.tor_row_indices,
            torch_module=self.torch,
            device=self.device,
        )
        stage1_logits = self.model100(flow100, dropout=float(self.config.target_model_dropout))
        stage2_logits = self.model700(flow700, dropout=float(self.config.target_model_dropout))
        return _mdeepcorr_cascade_logits(
            stage1_logits,
            stage2_logits,
            threshold=float(self.config.mdeepcorr_stage1_threshold),
            torch_module=self.torch,
        )

    def parameters(self):
        yield from self.model100.parameters()
        yield from self.model700.parameters()


class DeepCoFFEATorchTarget:
    _MODEL_FILE = "Deepcoffea.py"
    _DEFAULT_CHECKPOINT = (
        "deepcoffea/"
        "deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/"
        "best_loss.pth"
    )

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.torch = _import_torch()
        self.device = _torch_device(self.torch, config)
        root = Path(config.target_model_root)
        module = _load_module(root / self._MODEL_FILE, "work2_target_deepcoffea")
        self.anchor = module.Model(emb_size=64, input_size=config.deepcoffea_tor_len * 2).float().to(self.device)
        self.pandn = module.Model(emb_size=64, input_size=config.deepcoffea_exit_len * 2).float().to(self.device)
        checkpoint = _resolve_checkpoint(config, self._DEFAULT_CHECKPOINT)
        state_dict = _torch_load(self.torch, checkpoint, self.device)
        self.anchor.load_state_dict(state_dict["anchor_state_dict"])
        self.pandn.load_state_dict(state_dict["pandn_state_dict"])
        self.anchor.eval()
        self.pandn.eval()
        for module_parameter in list(self.anchor.parameters()) + list(self.pandn.parameters()):
            module_parameter.requires_grad = False

    def forward(self, adv_flow, exit_flow=None):
        if exit_flow is None:
            raise ValueError("DeepCoFFEA requires the paired exit flow; zero-filled exit inputs are not supported.")
        tor_input = _prepare_deepcoffea_flat_flow_torch(
            adv_flow=adv_flow,
            target_length=self.config.deepcoffea_tor_len,
            torch_module=self.torch,
            device=self.device,
        )
        exit_input = _prepare_deepcoffea_flat_flow_torch(
            adv_flow=exit_flow,
            target_length=self.config.deepcoffea_exit_len,
            torch_module=self.torch,
            device=self.device,
        )
        anchor_embedding = self.anchor(tor_input)
        exit_embedding = self.pandn(exit_input)
        if anchor_embedding.ndim == 1:
            anchor_embedding = anchor_embedding.unsqueeze(0)
        if exit_embedding.ndim == 1:
            exit_embedding = exit_embedding.unsqueeze(0)
        logits = self.torch.nn.functional.cosine_similarity(anchor_embedding, exit_embedding, dim=-1)
        return logits.reshape(-1, 1)

    def parameters(self):
        yield from self.anchor.parameters()
        yield from self.pandn.parameters()


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Real target model loading requires PyTorch to be installed.") from exc
    return torch


def _torch_device(torch_module, config: ExperimentConfig):
    if config.use_gpu and torch_module.cuda.is_available():
        return torch_module.device(config.target_device)
    return torch_module.device("cpu")


def _torch_load(torch_module, checkpoint: Path, device):
    try:
        return torch_module.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        return torch_module.load(checkpoint, map_location=device)


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    path = module_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Target model code not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load target model module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_checkpoint(config: ExperimentConfig, default_relative_path: str) -> Path:
    configured = str(config.target_model_path).strip()
    checkpoint = Path(configured) if configured else Path(config.target_model_root) / default_relative_path
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Target model checkpoint not found: {checkpoint}")
    return checkpoint


def _resolve_explicit_checkpoint(configured_path: str, default_path: Path, field_name: str) -> Path:
    checkpoint = Path(configured_path) if str(configured_path).strip() else default_path
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"{field_name} not found: {checkpoint}")
    return checkpoint


def _mdeepcorr_cascade_logits(stage1_logits, stage2_logits, threshold: float, torch_module):
    """Return logits whose sigmoid exactly matches Work1's two-stage score."""

    stage1_scores = torch_module.sigmoid(stage1_logits)
    stage2_scores = torch_module.sigmoid(stage2_logits)
    final_scores = torch_module.where(stage1_scores > threshold, stage2_scores, stage1_scores)
    epsilon = torch_module.finfo(final_scores.dtype).eps
    return torch_module.logit(final_scores.clamp(min=epsilon, max=1.0 - epsilon))


def _prepare_deepcorr_flow_torch(
    adv_flow,
    expected_length: int,
    tor_row_indices: list[int],
    torch_module,
    device,
):
    flow = adv_flow.to(device=device, dtype=torch_module.float32)
    if flow.ndim != 3:
        raise ValueError(f"DeepCorr target expects flow shape (batch, channels, length), got {tuple(flow.shape)}")
    if flow.shape[1] == 8:
        full_flow = flow
    elif flow.shape[1] == len(tor_row_indices):
        full_flow = torch_module.zeros(
            (flow.shape[0], 8, flow.shape[2]),
            dtype=flow.dtype,
            device=flow.device,
        )
        for source_index, target_row in enumerate(tor_row_indices):
            full_flow[:, int(target_row), :] = flow[:, source_index, :]
    else:
        raise ValueError(
            "DeepCorr target expects either 8-channel target flow or "
            f"{len(tor_row_indices)}-channel Tor flow, got {flow.shape[1]} channels."
        )

    if full_flow.shape[2] == expected_length:
        adjusted = full_flow
    elif full_flow.shape[2] > expected_length:
        adjusted = full_flow[:, :, :expected_length]
    else:
        adjusted = torch_module.zeros(
            (full_flow.shape[0], 8, expected_length),
            dtype=full_flow.dtype,
            device=full_flow.device,
        )
        adjusted[:, :, : full_flow.shape[2]] = full_flow

    return adjusted.unsqueeze(1)


def _prepare_deepcoffea_flat_flow_torch(adv_flow, target_length: int, torch_module, device):
    flow = adv_flow.to(device=device, dtype=torch_module.float32)
    flat_length = target_length * 2
    if flow.ndim >= 2 and flow.shape[-1] == flat_length:
        return flow.reshape(-1, flat_length)
    if flow.ndim != 3:
        raise ValueError(
            "DeepCoFFEA target expects (batch, 2, length) or pre-partitioned "
            f"(..., {flat_length}) windows, got {tuple(flow.shape)}"
        )
    if flow.shape[1] < 2:
        raise ValueError("DeepCoFFEA target expects at least ipd and size channels.")
    tor_flow = flow[:, :2, :]
    adjusted = torch_module.zeros(
        (flow.shape[0], 2, target_length),
        dtype=tor_flow.dtype,
        device=tor_flow.device,
    )
    copy_length = min(target_length, tor_flow.shape[2])
    adjusted[:, :, :copy_length] = tor_flow[:, :, :copy_length]
    return adjusted.reshape(flow.shape[0], target_length * 2)
