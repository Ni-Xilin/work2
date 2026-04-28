"""Torch-native second-workpoint model stack.

The torch-real path keeps generator-side trainable modules at one stable width
(``config.trainable_width``, falling back to ``config.d_model``) and bridges
into the frozen backbone embedding width with an explicit learned projector.
No numpy math appears on this main path.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

from second_workpoint.models.qwen_backbone import FrozenQwenBackbone

if TYPE_CHECKING:
    import torch

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None
    nn = None


def _require_torch():
    if torch is None or nn is None:
        raise ImportError("SecondWorkpointTorchRealModel requires PyTorch to be installed.")
    return torch, nn


def _model_width(config) -> int:
    return int(getattr(config, "trainable_width", getattr(config, "d_model", 0)))


if nn is not None:

    class TemporalPatchAdapter(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.config = config
            self.patch_len = int(config.patch_len)
            self.patch_num = int(config.seq_len // config.patch_len)
            input_dim = int(config.enc_in * config.patch_len)
            hidden_dim = _model_width(config)
            self.projection = nn.Linear(input_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, history_seq: "torch.Tensor") -> "torch.Tensor":
            batch_size = history_seq.shape[0]
            patch_view = history_seq.reshape(
                batch_size,
                self.config.enc_in,
                self.patch_num,
                self.patch_len,
            ).permute(0, 2, 1, 3)
            flat_inputs = patch_view.reshape(batch_size, self.patch_num, -1)
            return self.norm(self.projection(flat_inputs))


    class VisualPatchAdapter(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.config = config
            self.patch_len = int(config.patch_len)
            self.patch_num = int(config.seq_len // config.patch_len)
            input_dim = int(config.enc_in * config.visual_conv_channels)
            hidden_dim = _model_width(config)
            self.projection = nn.Linear(input_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, history_seq: "torch.Tensor") -> "torch.Tensor":
            batch_size = history_seq.shape[0]
            patch_view = history_seq.reshape(
                batch_size,
                self.config.enc_in,
                self.patch_num,
                self.patch_len,
            ).permute(0, 2, 1, 3)

            mean_feature = patch_view.mean(dim=-1)
            std_feature = patch_view.std(dim=-1, unbiased=False)
            max_feature = patch_view.max(dim=-1).values
            min_feature = patch_view.min(dim=-1).values
            stats = torch.stack([mean_feature, std_feature, max_feature, min_feature], dim=-1)

            if self.config.visual_conv_channels <= stats.shape[-1]:
                expanded = stats[:, :, :, : self.config.visual_conv_channels]
            else:
                repeat_factor = math.ceil(self.config.visual_conv_channels / stats.shape[-1])
                expanded = stats.repeat(1, 1, 1, repeat_factor)[:, :, :, : self.config.visual_conv_channels]

            flat_inputs = expanded.reshape(batch_size, self.patch_num, -1)
            return self.norm(self.projection(flat_inputs))


    class LearnedPrototypeReprogrammingLayer(nn.Module):
        """Legacy shared prototype bank kept for ablations and stub parity."""

        mode_name = "learned_prototypes"

        def __init__(self, num_prototypes: int, hidden_dim: int) -> None:
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            self.prototypes = nn.Parameter(torch.empty(int(num_prototypes), int(hidden_dim)))
            nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

        def build_prototype_state(self, _backbone: FrozenQwenBackbone | None = None) -> dict[str, "torch.Tensor | None"]:
            return {
                "prototype_bank_model": self.prototypes,
                "prototype_bank_backbone": None,
                "prototype_probe_entropy": None,
            }

        def route(
            self,
            tokens: "torch.Tensor",
            prototype_state: dict[str, "torch.Tensor | None"],
        ) -> dict[str, "torch.Tensor"]:
            prototypes = prototype_state["prototype_bank_model"]
            if prototypes is None:
                raise RuntimeError("learned prototype routing requires prototype_bank_model.")
            logits = torch.matmul(tokens, prototypes.transpose(0, 1)) / math.sqrt(self.hidden_dim)
            weights = torch.softmax(logits, dim=-1)
            reprogrammed = torch.matmul(weights, prototypes)
            return {
                "reprogrammed_tokens": reprogrammed,
                "routing_logits": logits,
                "routing_weights": weights,
            }


    class TextPrototypeReprogrammingLayer(nn.Module):
        """Time-LLM-style semantic alignment using E' = W E text prototypes."""

        mode_name = "text_prototypes"

        def __init__(
            self,
            num_prototypes: int,
            model_width: int,
            backbone_width: int,
            vocab_size: int,
            n_heads: int,
        ) -> None:
            super().__init__()
            self.num_prototypes = int(num_prototypes)
            self.model_width = int(model_width)
            self.backbone_width = int(backbone_width)
            self.n_heads = int(n_heads)
            self.head_dim = int(self.model_width // self.n_heads)

            self.prototype_probe_logits = nn.Parameter(torch.empty(self.num_prototypes, int(vocab_size)))
            nn.init.normal_(self.prototype_probe_logits, mean=0.0, std=0.02)

            self.query_projection = nn.Linear(self.model_width, self.model_width)
            self.key_projection = nn.Linear(self.backbone_width, self.model_width)
            self.value_projection = nn.Linear(self.backbone_width, self.model_width)
            self.output_norm = nn.LayerNorm(self.model_width)

        def build_prototype_state(self, backbone: FrozenQwenBackbone) -> dict[str, "torch.Tensor | None"]:
            probe_weights = torch.softmax(self.prototype_probe_logits.float(), dim=-1)
            prototype_bank_backbone = backbone.build_text_prototypes(probe_weights)
            prototype_bank_model = prototype_bank_backbone.to(
                device=self.query_projection.weight.device,
                dtype=self.query_projection.weight.dtype,
            )
            probe_entropy = -(
                probe_weights * probe_weights.clamp_min(1e-9).log()
            ).sum(dim=-1).mean()
            return {
                "prototype_bank_model": prototype_bank_model,
                "prototype_bank_backbone": prototype_bank_backbone,
                "prototype_probe_entropy": probe_entropy,
            }

        def route(
            self,
            tokens: "torch.Tensor",
            prototype_state: dict[str, "torch.Tensor | None"],
        ) -> dict[str, "torch.Tensor"]:
            prototype_bank = prototype_state["prototype_bank_model"]
            if prototype_bank is None:
                raise RuntimeError("text prototype routing requires prototype_bank_model.")

            query = self.query_projection(tokens)
            key = self.key_projection(prototype_bank)
            value = self.value_projection(prototype_bank)

            batch_size, token_count, _ = query.shape
            query = query.reshape(batch_size, token_count, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            key = key.reshape(self.num_prototypes, self.n_heads, self.head_dim).permute(1, 0, 2)
            value = value.reshape(self.num_prototypes, self.n_heads, self.head_dim).permute(1, 0, 2)

            logits = torch.einsum("bhpd,hvd->bhpv", query, key) / math.sqrt(self.head_dim)
            weights = torch.softmax(logits, dim=-1)
            attended = torch.einsum("bhpv,hvd->bhpd", weights, value)
            attended = attended.permute(0, 2, 1, 3).reshape(batch_size, token_count, self.model_width)

            return {
                "reprogrammed_tokens": self.output_norm(attended),
                "routing_logits": logits.mean(dim=1),
                "routing_weights": weights.mean(dim=1),
            }


    class BackboneWidthBridge(nn.Module):
        """Explicit width seam between stable trainable width and frozen backbone width."""

        def __init__(self, model_width: int, backbone_width: int) -> None:
            super().__init__()
            self.model_width = int(model_width)
            self.backbone_width = int(backbone_width)
            self.to_backbone_norm = nn.LayerNorm(self.model_width)
            self.to_backbone = nn.Linear(self.model_width, self.backbone_width)
            self.from_backbone_norm = nn.LayerNorm(self.backbone_width)
            self.from_backbone = nn.Linear(self.backbone_width, self.model_width)

        def project_to_backbone(self, tokens: "torch.Tensor") -> "torch.Tensor":
            return self.to_backbone(self.to_backbone_norm(tokens))

        def project_from_backbone(self, context_tokens: "torch.Tensor") -> "torch.Tensor":
            return self.from_backbone(self.from_backbone_norm(context_tokens.float()))


    class PerturbationHead(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            hidden_dim = _model_width(config)
            output_dim = int(config.pred_len * config.enc_in)
            self.pred_len = int(config.pred_len)
            self.enc_in = int(config.enc_in)
            self.norm = nn.LayerNorm(hidden_dim)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.activation = nn.GELU()

        def forward(self, context_tokens: "torch.Tensor", token_mask: "torch.Tensor | None" = None) -> "torch.Tensor":
            if token_mask is None:
                pooled_context = context_tokens.mean(dim=1)
            else:
                mask = token_mask.to(device=context_tokens.device, dtype=context_tokens.dtype).unsqueeze(-1)
                pooled_context = (context_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            hidden = self.activation(self.fc1(self.norm(pooled_context)))
            perturbation = self.fc2(hidden)
            return perturbation.reshape(context_tokens.shape[0], self.pred_len, self.enc_in)


    class SecondWorkpointTorchRealModel(nn.Module):
        """Torch-native second-workpoint model for the real frozen-backbone path."""

        def __init__(self, config) -> None:
            super().__init__()
            torch_module, _ = _require_torch()
            self.config = config
            self.semantic_alignment_mode = str(getattr(config, "semantic_alignment_mode", "learned_prototypes"))
            self.temporal_adapter = TemporalPatchAdapter(config)
            self.visual_adapter = VisualPatchAdapter(config)
            self.backbone = FrozenQwenBackbone(config)
            if self.semantic_alignment_mode == "text_prototypes":
                self.shared_reprogramming = TextPrototypeReprogrammingLayer(
                    num_prototypes=int(config.reprogramming_num_prototypes),
                    model_width=_model_width(config),
                    backbone_width=int(self.backbone.embedding_width),
                    vocab_size=int(self.backbone.vocab_size),
                    n_heads=int(config.n_heads),
                )
            elif self.semantic_alignment_mode == "learned_prototypes":
                self.shared_reprogramming = LearnedPrototypeReprogrammingLayer(
                    num_prototypes=int(config.reprogramming_num_prototypes),
                    hidden_dim=_model_width(config),
                )
            else:
                raise ValueError(
                    "Unsupported semantic_alignment_mode on torch_real path: "
                    f"{self.semantic_alignment_mode}"
                )
            self.width_bridge = BackboneWidthBridge(
                model_width=_model_width(config),
                backbone_width=int(self.backbone.embedding_width),
            )
            self.output_head = PerturbationHead(config)
            self.backbone_device = self.backbone.device
            self.trainable_device = torch_module.device(str(getattr(config, "trainable_device", self.backbone_device)))
            for module in (
                self.temporal_adapter,
                self.visual_adapter,
                self.shared_reprogramming,
                self.width_bridge,
                self.output_head,
            ):
                module.to(self.trainable_device)

        def train(self, mode: bool = True):
            super().train(mode)
            self.backbone.eval()
            return self

        def _resolve_device(self) -> "torch.device":
            return self.trainable_device

        def forward(
            self,
            history_seq: "torch.Tensor",
            prompt_text: Sequence[str] | None = None,
            prompt_ids: "torch.Tensor | None" = None,
        ) -> dict[str, "torch.Tensor | dict[str, int]"]:
            if prompt_ids is not None:
                raise ValueError("prompt_ids are forbidden on the torch_real model path; pass prompt_text instead.")
            if prompt_text is None:
                raise ValueError("prompt_text is required on the torch_real model path.")

            device = self._resolve_device()
            history_seq = history_seq.to(device=device, dtype=torch.float32)

            temporal_tokens = self.temporal_adapter(history_seq)
            visual_tokens = self.visual_adapter(history_seq)
            prototype_state = self.shared_reprogramming.build_prototype_state(self.backbone)
            temporal_route = self.shared_reprogramming.route(temporal_tokens, prototype_state)
            visual_route = self.shared_reprogramming.route(visual_tokens, prototype_state)

            reprogrammed_temporal = temporal_route["reprogrammed_tokens"]
            reprogrammed_visual = visual_route["reprogrammed_tokens"]

            backbone_temporal = self.width_bridge.project_to_backbone(reprogrammed_temporal)
            backbone_visual = self.width_bridge.project_to_backbone(reprogrammed_visual)
            backbone_outputs = self.backbone(
                prompt_text=prompt_text,
                time_tokens=backbone_temporal,
                visual_tokens=backbone_visual,
            )
            backbone_context = backbone_outputs["last_hidden_state"]
            conditioning_mask = backbone_outputs["conditioning_mask"]
            generator_context = self.width_bridge.project_from_backbone(
                backbone_context.to(self.trainable_device)
            )
            perturbation = self.output_head(
                generator_context,
                token_mask=conditioning_mask.to(self.trainable_device),
            )

            return {
                "temporal_tokens": temporal_tokens,
                "visual_tokens": visual_tokens,
                "reprogrammed_temporal": reprogrammed_temporal,
                "reprogrammed_visual": reprogrammed_visual,
                "backbone_temporal_tokens": backbone_temporal,
                "backbone_visual_tokens": backbone_visual,
                "semantic_alignment_mode": self.semantic_alignment_mode,
                "prototype_bank_model": prototype_state["prototype_bank_model"],
                "prototype_bank_backbone": prototype_state["prototype_bank_backbone"],
                "prototype_weights_temporal": temporal_route["routing_weights"],
                "prototype_weights_visual": visual_route["routing_weights"],
                "prototype_logits_temporal": temporal_route["routing_logits"],
                "prototype_logits_visual": visual_route["routing_logits"],
                "context_tokens": generator_context,
                "backbone_context_tokens": backbone_context,
                "conditioning_mask": conditioning_mask,
                "perturbation": perturbation,
                "shape_contract": {
                    "model_width": int(self.width_bridge.model_width),
                    "backbone_width": int(self.width_bridge.backbone_width),
                    "num_prototypes": int(self.config.reprogramming_num_prototypes),
                    "conditioning_token_count": int(backbone_temporal.shape[1] + backbone_visual.shape[1]),
                },
                "device_contract": {
                    "trainable_device": str(self.trainable_device),
                    "backbone_device": str(self.backbone_device),
                },
                "attention_stats": {
                    "temporal_weight_mean": temporal_route["routing_weights"].mean(),
                    "visual_weight_mean": visual_route["routing_weights"].mean(),
                    "prototype_probe_entropy": prototype_state["prototype_probe_entropy"],
                },
            }

else:

    class SecondWorkpointTorchRealModel:  # pragma: no cover - exercised only without torch
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()
