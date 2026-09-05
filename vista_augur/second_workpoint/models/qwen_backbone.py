"""Frozen Hugging Face Qwen backbone wrapper for the second work point.

This module keeps the backbone frozen while preserving autograd through
``inputs_embeds`` so upstream trainable adapters can learn against the real
Qwen hidden-state space.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from second_workpoint.utils.runtime import resolve_torch_dtype

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
        raise ImportError("FrozenQwenBackbone requires PyTorch to be installed.")
    return torch, nn


def _require_transformers():
    try:
        import transformers
        from transformers import AutoModel, AutoTokenizer
        from transformers.models.qwen2.modeling_qwen2 import create_causal_mask, create_sliding_window_causal_mask
    except ImportError as exc:  # pragma: no cover - exercised in environments without transformers
        raise ImportError("FrozenQwenBackbone requires transformers to be installed.") from exc
    return transformers, AutoModel, AutoTokenizer, create_causal_mask, create_sliding_window_causal_mask


def _prefers_dtype_argument(transformers_module) -> bool:
    version_text = str(getattr(transformers_module, "__version__", "0.0.0"))
    version_core = version_text.split("+", 1)[0]
    parts = version_core.split(".")
    numbers: list[int] = []
    for part in parts[:3]:
        digits = []
        for char in part:
            if char.isdigit():
                digits.append(char)
            else:
                break
        numbers.append(int("".join(digits) or "0"))
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers) >= (4, 56, 0)


def _resolve_model_source(config) -> str:
    explicit_path = str(getattr(config, "backbone_model_path", "")).strip()
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Configured backbone_model_path does not exist: {candidate}")

    explicit_name = str(getattr(config, "backbone_model_name", "")).strip()
    if not explicit_name:
        raise ValueError("A real Qwen backbone requires backbone_model_path or backbone_model_name.")
    candidate = Path(explicit_name)
    if candidate.exists():
        return str(candidate)
    return explicit_name


def _resolve_device(config) -> "torch.device":
    torch_module, _ = _require_torch()
    explicit_device = getattr(config, "backbone_device", None)
    if explicit_device:
        return torch_module.device(str(explicit_device))
    if getattr(config, "use_gpu", False) and torch_module.cuda.is_available():
        return torch_module.device(f"cuda:{getattr(config, 'gpu', 0)}")
    return torch_module.device("cpu")


if nn is not None:

    class FrozenQwenBackbone(nn.Module):
        """Frozen Qwen-family backbone with tokenizer + ``inputs_embeds`` path."""

        def __init__(self, config) -> None:
            super().__init__()
            torch_module, _ = _require_torch()
            transformers_module, AutoModel, AutoTokenizer, create_causal_mask, create_sliding_window_causal_mask = _require_transformers()

            self.config = config
            self.model_name = _resolve_model_source(config)
            self.device = _resolve_device(config)
            self.secondary_device = None
            explicit_secondary = str(getattr(config, "backbone_secondary_device", "")).strip()
            if explicit_secondary:
                self.secondary_device = torch_module.device(explicit_secondary)
            self.max_prompt_tokens = int(getattr(config, "backbone_prompt_max_tokens", 0))
            self.split_layer_index = int(getattr(config, "backbone_split_layer_index", 0))
            self.model_dtype = resolve_torch_dtype(str(getattr(config, "backbone_dtype", "float16")))
            self.quantization = str(getattr(config, "backbone_quantization", "none"))
            self._create_causal_mask = create_causal_mask
            self._create_sliding_window_causal_mask = create_sliding_window_causal_mask

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            dtype_argument = "dtype" if _prefers_dtype_argument(transformers_module) else "torch_dtype"
            model_kwargs[dtype_argument] = self.model_dtype
            if self.quantization != "none":
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError as exc:
                    raise ImportError("backbone quantization requires transformers BitsAndBytesConfig support.") from exc
                try:
                    import bitsandbytes  # noqa: F401
                except ImportError as exc:
                    raise ImportError(
                        "backbone_quantization was requested but bitsandbytes is not installed in the active environment."
                    ) from exc
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=self.quantization == "4bit",
                    load_in_8bit=self.quantization == "8bit",
                )
                if self.device.type == "cuda":
                    model_kwargs["device_map"] = {"": self.device.index if self.device.index is not None else 0}

            self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
            self.model.eval()
            if self.quantization == "none":
                self.model.to(self.device)
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

            self.core_model = getattr(self.model, "model", self.model)
            self.split_across_devices = self._configure_split_devices()

            embedding_layer = self.model.get_input_embeddings()
            self.embedding_width = int(embedding_layer.embedding_dim)
            self.vocab_size = int(embedding_layer.num_embeddings)
            self.backbone_dtype = embedding_layer.weight.dtype
            self._torch = torch_module

        def _configure_split_devices(self) -> bool:
            if self.secondary_device is None:
                return False
            if self.secondary_device == self.device:
                return False
            if self.quantization != "none":
                return False
            if self.split_layer_index <= 0:
                return False
            if not hasattr(self.core_model, "layers"):
                return False

            layer_count = len(self.core_model.layers)
            if self.split_layer_index >= layer_count:
                raise ValueError(
                    f"backbone_split_layer_index must be < layer_count ({layer_count}), got {self.split_layer_index}."
                )

            for layer in self.core_model.layers[self.split_layer_index :]:
                layer.to(self.secondary_device)
            if hasattr(self.core_model, "norm"):
                self.core_model.norm.to(self.secondary_device)
            return True

        def _move_position_embeddings(self, position_embeddings, device):
            if isinstance(position_embeddings, tuple):
                return tuple(component.to(device) for component in position_embeddings)
            return position_embeddings.to(device)

        def _build_attention_masks(self, inputs_embeds, attention_mask, position_ids, cache_position, past_key_values):
            if isinstance(attention_mask, dict):
                return attention_mask
            mask_kwargs = {
                "config": self.core_model.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": self._create_causal_mask(**mask_kwargs),
            }
            if getattr(self.core_model, "has_sliding_layers", False):
                causal_mask_mapping["sliding_attention"] = self._create_sliding_window_causal_mask(**mask_kwargs)
            return causal_mask_mapping

        def _forward_split_backbone(self, context_embeddings, attention_mask):
            past_key_values = None
            cache_position = self._torch.arange(context_embeddings.shape[1], device=context_embeddings.device)
            position_ids = cache_position.unsqueeze(0)
            causal_mask_mapping = self._build_attention_masks(
                inputs_embeds=context_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past_key_values,
            )
            hidden_states = context_embeddings
            position_embeddings = self.core_model.rotary_emb(hidden_states, position_ids)

            for decoder_layer in self.core_model.layers[: self.split_layer_index]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = hidden_states.to(self.secondary_device)
            position_ids_secondary = position_ids.to(self.secondary_device)
            cache_position_secondary = cache_position.to(self.secondary_device)
            position_embeddings_secondary = self._move_position_embeddings(position_embeddings, self.secondary_device)
            causal_mask_mapping_secondary = {
                key: (value.to(self.secondary_device) if value is not None else None)
                for key, value in causal_mask_mapping.items()
            }
            for decoder_layer in self.core_model.layers[self.split_layer_index :]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping_secondary[decoder_layer.attention_type],
                    position_ids=position_ids_secondary,
                    past_key_values=past_key_values,
                    use_cache=False,
                    cache_position=cache_position_secondary,
                    position_embeddings=position_embeddings_secondary,
                )
            hidden_states = self.core_model.norm(hidden_states)
            return hidden_states

        def get_input_embedding_weight(self) -> "torch.Tensor":
            return self.model.get_input_embeddings().weight

        def build_text_prototypes(self, probe_weights: "torch.Tensor") -> "torch.Tensor":
            if probe_weights.ndim != 2:
                raise ValueError("probe_weights must be a rank-2 tensor shaped [num_prototypes, vocab_size].")
            embedding_weight = self.get_input_embedding_weight()
            if probe_weights.shape[-1] != embedding_weight.shape[0]:
                raise ValueError(
                    "probe_weights vocab dimension does not match backbone embedding table: "
                    f"{probe_weights.shape[-1]} != {embedding_weight.shape[0]}"
                )
            probe_weights = probe_weights.to(device=self.device, dtype=embedding_weight.dtype)
            return self._torch.matmul(probe_weights, embedding_weight)

        def tokenize_prompt(self, prompt_text: Sequence[str]) -> tuple["torch.Tensor", "torch.Tensor"]:
            encoded = self.tokenizer(
                list(prompt_text),
                padding=True,
                truncation=self.max_prompt_tokens > 0,
                max_length=self.max_prompt_tokens if self.max_prompt_tokens > 0 else None,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            return input_ids, attention_mask

        def encode_prompt(self, prompt_text: Sequence[str]) -> tuple["torch.Tensor", "torch.Tensor"]:
            input_ids, attention_mask = self.tokenize_prompt(prompt_text)
            prompt_embeddings = self.model.get_input_embeddings()(input_ids)
            return prompt_embeddings, attention_mask

        def forward(
            self,
            prompt_text: Sequence[str],
            time_tokens: "torch.Tensor",
            visual_tokens: "torch.Tensor",
        ) -> dict[str, "torch.Tensor"]:
            if not prompt_text:
                raise ValueError("FrozenQwenBackbone requires non-empty prompt_text.")
            prompt_embeddings, prompt_mask = self.encode_prompt(prompt_text)

            time_tokens = time_tokens.to(device=self.device, dtype=self.backbone_dtype)
            visual_tokens = visual_tokens.to(device=self.device, dtype=self.backbone_dtype)
            prompt_embeddings = prompt_embeddings.to(dtype=self.backbone_dtype)

            context_embeddings = self._torch.cat([prompt_embeddings, time_tokens, visual_tokens], dim=1)
            token_mask = self._torch.ones(
                (context_embeddings.shape[0], time_tokens.shape[1] + visual_tokens.shape[1]),
                dtype=prompt_mask.dtype,
                device=self.device,
            )
            attention_mask = self._torch.cat([prompt_mask, token_mask], dim=1)
            if self.split_across_devices:
                last_hidden_state = self._forward_split_backbone(context_embeddings, attention_mask)
            else:
                outputs = self.model(
                    inputs_embeds=context_embeddings,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                if not hasattr(outputs, "last_hidden_state"):
                    raise RuntimeError("Qwen backbone did not return last_hidden_state.")
                last_hidden_state = outputs.last_hidden_state
            conditioning_mask = self._torch.cat(
                [
                    self._torch.zeros_like(prompt_mask),
                    token_mask,
                ],
                dim=1,
            )
            return {
                "last_hidden_state": last_hidden_state,
                "attention_mask": attention_mask,
                "conditioning_mask": conditioning_mask,
                "prompt_attention_mask": prompt_mask,
            }

else:

    class FrozenQwenBackbone:  # pragma: no cover - exercised only without torch
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()
