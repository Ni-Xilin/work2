"""Target model factory for the second work point.

This module serves both the legacy stub path and the new ``torch_real`` path:
keep the default stub for fast tests, or set ``target_model_mode=torch`` in
the config to load the real top-level ``target_model`` code and weights.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from second_workpoint.config import ExperimentConfig
from second_workpoint.models.target_model_stub import FrozenTargetModelStub


def build_target_model(config: ExperimentConfig):
    if config.target_model_mode == "stub":
        return FrozenTargetModelStub(config)
    return build_torch_target_model(config)


def build_torch_target_model(config: ExperimentConfig):
    target_name = config.target_model.lower()
    if target_name in {"deepcorr300", "deepcorr300torch"}:
        return DeepCorrTorchTarget(config, variant="300")
    if target_name in {"deepcorr100", "deepcorr100torch"}:
        return DeepCorrTorchTarget(config, variant="100")
    if target_name in {"deepcorr700", "mdeepcorr", "mdeepcorrtorch"}:
        return DeepCorrTorchTarget(config, variant="700")
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

    def forward(self, adv_flow):
        if _is_torch_tensor(adv_flow):
            flow = _prepare_deepcorr_flow_torch(
                adv_flow=adv_flow,
                expected_length=self._EXPECTED_LENGTHS[self.variant],
                tor_row_indices=self.config.tor_row_indices,
                torch_module=self.torch,
                device=self.device,
            )
            return self.model(flow, dropout=float(self.config.target_model_dropout))

        flow = _prepare_deepcorr_flow_numpy(
            adv_flow=adv_flow,
            expected_length=self._EXPECTED_LENGTHS[self.variant],
            tor_row_indices=self.config.tor_row_indices,
        )
        tensor = self.torch.from_numpy(flow).to(self.device)
        logits = self.model(tensor, dropout=float(self.config.target_model_dropout))
        return logits.detach().cpu().numpy().astype(np.float32)

    def parameters(self):
        return self.model.parameters()


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

    def forward(self, adv_flow):
        if _is_torch_tensor(adv_flow):
            tor_input = _prepare_deepcoffea_flat_flow_torch(
                adv_flow=adv_flow,
                target_length=self.config.deepcoffea_tor_len,
                torch_module=self.torch,
                device=self.device,
            )
            exit_input = self.torch.zeros(
                (tor_input.shape[0], self.config.deepcoffea_exit_len * 2),
                dtype=tor_input.dtype,
                device=tor_input.device,
            )
            anchor_embedding = self.anchor(tor_input)
            exit_embedding = self.pandn(exit_input)
            if anchor_embedding.ndim == 1:
                anchor_embedding = anchor_embedding.unsqueeze(0)
            if exit_embedding.ndim == 1:
                exit_embedding = exit_embedding.unsqueeze(0)
            logits = self.torch.nn.functional.cosine_similarity(anchor_embedding, exit_embedding, dim=-1)
            return logits.reshape(-1, 1)

        tor_input = _prepare_deepcoffea_flat_flow_numpy(adv_flow, self.config.deepcoffea_tor_len)
        exit_input = np.zeros((tor_input.shape[0], self.config.deepcoffea_exit_len * 2), dtype=np.float32)
        tor_tensor = self.torch.from_numpy(tor_input).to(self.device)
        exit_tensor = self.torch.from_numpy(exit_input).to(self.device)
        anchor_embedding = self.anchor(tor_tensor)
        exit_embedding = self.pandn(exit_tensor)
        if anchor_embedding.ndim == 1:
            anchor_embedding = anchor_embedding.unsqueeze(0)
        if exit_embedding.ndim == 1:
            exit_embedding = exit_embedding.unsqueeze(0)
        logits = self.torch.nn.functional.cosine_similarity(anchor_embedding, exit_embedding, dim=-1)
        return logits.reshape(-1, 1).detach().cpu().numpy().astype(np.float32)

    def parameters(self):
        yield from self.anchor.parameters()
        yield from self.pandn.parameters()


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("target_model_mode=torch requires PyTorch to be installed.") from exc
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
    configured = Path(config.target_model_path)
    if str(configured) and str(configured) != "work2/outputs/target_stub":
        checkpoint = configured
    else:
        checkpoint = Path(config.target_model_root) / default_relative_path
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Target model checkpoint not found: {checkpoint}")
    return checkpoint


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "device") and hasattr(value, "requires_grad")


def _prepare_deepcorr_flow_numpy(
    adv_flow: np.ndarray,
    expected_length: int,
    tor_row_indices: list[int],
) -> np.ndarray:
    flow = np.asarray(adv_flow, dtype=np.float32)
    if flow.ndim != 3:
        raise ValueError(f"DeepCorr target expects flow shape (batch, channels, length), got {flow.shape}")
    if flow.shape[1] == 8:
        full_flow = flow
    elif flow.shape[1] == len(tor_row_indices):
        full_flow = np.zeros((flow.shape[0], 8, flow.shape[2]), dtype=np.float32)
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
        adjusted = np.zeros((full_flow.shape[0], 8, expected_length), dtype=np.float32)
        adjusted[:, :, : full_flow.shape[2]] = full_flow

    return adjusted[:, None, :, :].astype(np.float32)


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


def _prepare_deepcoffea_flat_flow_numpy(adv_flow: np.ndarray, target_length: int) -> np.ndarray:
    flow = np.asarray(adv_flow, dtype=np.float32)
    if flow.ndim != 3:
        raise ValueError(f"DeepCoFFEA target expects flow shape (batch, channels, length), got {flow.shape}")
    if flow.shape[1] < 2:
        raise ValueError("DeepCoFFEA target expects at least ipd and size channels.")
    tor_flow = flow[:, :2, :]
    adjusted = np.zeros((flow.shape[0], 2, target_length), dtype=np.float32)
    copy_length = min(target_length, tor_flow.shape[2])
    adjusted[:, :, :copy_length] = tor_flow[:, :, :copy_length]
    return adjusted.reshape(flow.shape[0], target_length * 2).astype(np.float32)


def _prepare_deepcoffea_flat_flow_torch(adv_flow, target_length: int, torch_module, device):
    flow = adv_flow.to(device=device, dtype=torch_module.float32)
    if flow.ndim != 3:
        raise ValueError(f"DeepCoFFEA target expects flow shape (batch, channels, length), got {tuple(flow.shape)}")
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
