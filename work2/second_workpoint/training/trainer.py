"""第二工作点真实训练器。

这个文件只保留真实 ``torch_real`` 主路径：
- 真实 dataloader
- 真实 frozen Qwen backbone
- 真实 frozen target model
- torch autograd / optimizer / checkpoint
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from second_workpoint.config import ExperimentConfig
from second_workpoint.data.factory import build_dataset
from second_workpoint.models.target_model_adapter import build_target_model
from second_workpoint.training.losses import TargetedOverheadLoss
from second_workpoint.utils.runtime import count_parameters, ensure_dir, import_torch, resolve_device, resolve_torch_device, save_json


def build_trainer(config: ExperimentConfig):
    return TorchTrainer(config)


class TorchTrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.torch = import_torch()
        self.device = resolve_device(config)
        self.trainable_device = resolve_torch_device(config.trainable_device)
        self.setting_name = config.setting_name()
        self.run_dir = ensure_dir(Path(config.checkpoints) / self.setting_name)

        self.train_dataset = build_dataset(config, split="train")
        self.eval_dataset = build_dataset(config, split="eval")

        from torch.utils.data import DataLoader
        from second_workpoint.models import SecondWorkpointTorchRealModel

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=self._torch_collate_batch,
            drop_last=False,
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=self._torch_collate_batch,
            drop_last=False,
        )

        self.model = SecondWorkpointTorchRealModel(config)
        self.target_model = build_target_model(config)
        self.criterion = TargetedOverheadLoss(config)
        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = self.torch.optim.Adam(trainable_parameters, lr=config.learning_rate)

        self._log_model_summary()

    def train(self) -> dict:
        history: list[dict] = []
        for epoch in range(1, self.config.train_epochs + 1):
            self.model.train()
            epoch_metrics = {
                "loss": 0.0,
                "label_loss": 0.0,
                "time_ratio": 0.0,
                "size_ratio": 0.0,
            }
            steps = 0
            self.optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(self.train_loader, start=1):
                outputs = self._forward_batch(batch, log_shapes=(epoch == 1 and step == 1))
                scaled_loss = outputs["loss"] / float(self.config.gradient_accumulation_steps)
                scaled_loss.backward()

                reached_train_step_limit = self.config.max_train_steps > 0 and (steps + 1) >= self.config.max_train_steps
                should_step = (
                    (step % self.config.gradient_accumulation_steps == 0)
                    or (step == len(self.train_loader))
                    or reached_train_step_limit
                )
                if should_step:
                    if self.config.max_grad_norm > 0:
                        self.torch.nn.utils.clip_grad_norm_(
                            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                            max_norm=float(self.config.max_grad_norm),
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                epoch_metrics["loss"] += float(outputs["loss"].detach().item())
                epoch_metrics["label_loss"] += float(outputs["label_loss"])
                epoch_metrics["time_ratio"] += float(outputs["time_ratio"])
                epoch_metrics["size_ratio"] += float(outputs["size_ratio"])
                steps += 1

                if step % self.config.log_interval == 0:
                    print(
                        f"[train] epoch={epoch} step={step} "
                        f"loss={float(outputs['loss'].detach().item()):.4f} "
                        f"label={outputs['label_loss']:.4f} "
                        f"time={outputs['time_ratio']:.4f} "
                        f"size={outputs['size_ratio']:.4f}"
                    )
                if self.config.max_train_steps > 0 and steps >= self.config.max_train_steps:
                    break

            train_summary = self._average_metrics(epoch_metrics, steps)
            eval_summary = self.evaluate()
            history.append({"epoch": epoch, "train": train_summary, "eval": eval_summary})
            print(f"[epoch] {epoch} train_loss={train_summary['loss']:.4f} eval_loss={eval_summary['loss']:.4f}")
            self._save_checkpoint(epoch, train_summary, eval_summary)

        summary = {
            "setting_name": self.setting_name,
            "device": self.device,
            "backend": self.config.backend,
            "epochs": history,
        }
        save_json(self.run_dir / "train_summary.json", summary)
        return summary

    def evaluate(self) -> dict:
        self.model.eval()
        metrics = {
            "loss": 0.0,
            "label_loss": 0.0,
            "time_ratio": 0.0,
            "size_ratio": 0.0,
        }
        steps = 0
        with self.torch.no_grad():
            for step, batch in enumerate(self.eval_loader, start=1):
                outputs = self._forward_batch(batch, log_shapes=False)
                metrics["loss"] += float(outputs["loss"].detach().item())
                metrics["label_loss"] += float(outputs["label_loss"])
                metrics["time_ratio"] += float(outputs["time_ratio"])
                metrics["size_ratio"] += float(outputs["size_ratio"])
                steps += 1
                if self.config.max_eval_steps > 0 and steps >= self.config.max_eval_steps:
                    break
        return self._average_metrics(metrics, steps)

    def _torch_collate_batch(self, samples):
        batch = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            first_value = values[0]
            if isinstance(first_value, str):
                batch[key] = [str(value) for value in values]
                continue
            if isinstance(first_value, np.ndarray):
                if first_value.ndim == 0 and first_value.dtype.kind in {"U", "S", "O"}:
                    batch[key] = [str(value.item()) for value in values]
                    continue
                stacked = np.stack(values, axis=0)
                tensor = self.torch.from_numpy(stacked)
                if np.issubdtype(stacked.dtype, np.floating):
                    tensor = tensor.float()
                batch[key] = tensor
                continue
            batch[key] = values
        return batch

    def _forward_batch(self, batch, log_shapes: bool):
        full_flow = batch["full_flow"].to(self.trainable_device, dtype=self.torch.float32)
        history_seq = batch["history_seq"].to(self.trainable_device, dtype=self.torch.float32)
        clean_future = batch["clean_future"].to(self.trainable_device, dtype=self.torch.float32)
        prompt_text = list(batch["prompt_text"])
        future_mask = (
            batch["future_mask"].to(self.trainable_device, dtype=self.torch.float32)
            if "future_mask" in batch
            else self.torch.ones((full_flow.shape[0], self.config.pred_len), device=self.trainable_device)
        )
        writeback_meta = (
            batch["writeback_meta"].to(self.trainable_device, dtype=self.torch.long)
            if "writeback_meta" in batch
            else self.torch.tensor(
                [[self.config.seq_len, self.config.pred_len] for _ in range(full_flow.shape[0])],
                device=self.trainable_device,
                dtype=self.torch.long,
            )
        )
        future_mask_3d = future_mask.unsqueeze(-1)

        model_outputs = self.model.forward(history_seq=history_seq, prompt_text=prompt_text)
        perturbation = model_outputs["perturbation"]
        effective_perturbation = perturbation * future_mask_3d
        positive_delta = effective_perturbation.abs()
        direction = self.torch.where(clean_future >= 0.0, self.torch.ones_like(clean_future), -self.torch.ones_like(clean_future))
        adv_future = direction * (clean_future.abs() + positive_delta)

        adv_flow = full_flow.clone()
        flow_mask = self.torch.ones((full_flow.shape[0], full_flow.shape[2]), dtype=self.torch.float32, device=self.trainable_device)
        target_original_flow = (
            batch["target_full_flow"].to(self.trainable_device, dtype=self.torch.float32).clone()
            if "target_full_flow" in batch
            else full_flow.clone()
        )
        target_adv_flow = target_original_flow.clone()

        for sample_index in range(full_flow.shape[0]):
            writeback_start = int(writeback_meta[sample_index, 0].item())
            valid_length = int(writeback_meta[sample_index, 1].item())
            flow_mask[sample_index, writeback_start : writeback_start + self.config.pred_len] = future_mask[sample_index]
            if valid_length <= 0:
                continue
            writeback_end = writeback_start + valid_length
            adv_flow[sample_index, :, writeback_start:writeback_end] = adv_future[sample_index, :valid_length, :].transpose(0, 1)
            self._write_target_future_torch(
                target_adv_flow=target_adv_flow,
                sample_index=sample_index,
                writeback_start=writeback_start,
                valid_length=valid_length,
                adv_future=adv_future,
            )

        target_logits = self.target_model.forward(target_adv_flow).to(self.trainable_device)
        target_labels = self.torch.zeros_like(target_logits, dtype=self.torch.float32, device=self.trainable_device)
        loss_outputs = self.criterion.forward(
            target_logits=target_logits,
            target_labels=target_labels,
            original_flow=full_flow,
            adv_flow=adv_flow,
            flow_mask=flow_mask,
        )

        if log_shapes:
            print(
                "[shape] "
                f"history_seq={tuple(history_seq.shape)} "
                f"prompt_count={len(prompt_text)} "
                f"perturbation={tuple(perturbation.shape)} "
                f"adv_flow={tuple(adv_flow.shape)} "
                f"target_adv_flow={tuple(target_adv_flow.shape)} "
                f"logits={tuple(target_logits.shape)}"
            )

        return {
            "loss": loss_outputs["loss"],
            "label_loss": float(loss_outputs["label_loss"].detach().item()),
            "time_ratio": float(loss_outputs["time_ratio"].detach().item()),
            "size_ratio": float(loss_outputs["size_ratio"].detach().item()),
        }

    def _write_target_future_torch(self, target_adv_flow, sample_index: int, writeback_start: int, valid_length: int, adv_future) -> None:
        writeback_end = writeback_start + valid_length
        future_slice = adv_future[sample_index, :valid_length, :].transpose(0, 1)
        if target_adv_flow.shape[1] == future_slice.shape[0]:
            target_adv_flow[sample_index, :, writeback_start:writeback_end] = future_slice
            return
        if target_adv_flow.shape[1] == 8 and future_slice.shape[0] == len(self.config.tor_row_indices):
            for source_index, target_row in enumerate(self.config.tor_row_indices):
                target_adv_flow[sample_index, int(target_row), writeback_start:writeback_end] = future_slice[source_index]
            return
        raise ValueError(
            "Cannot write generated future into target flow: "
            f"target_channels={target_adv_flow.shape[1]} generated_channels={future_slice.shape[0]}"
        )

    def _save_checkpoint(self, epoch: int, train_summary: dict, eval_summary: dict) -> None:
        checkpoint_prefix = self.run_dir / f"checkpoint_epoch_{epoch}"
        save_json(
            checkpoint_prefix.with_suffix(".json"),
            {
                "epoch": epoch,
                "config": self.config.to_dict(),
                "train_summary": train_summary,
                "eval_summary": eval_summary,
            },
        )
        self.torch.save(
            {
                "epoch": epoch,
                "config": self.config.to_dict(),
                "train_summary": train_summary,
                "eval_summary": eval_summary,
                "model_state": self._trainable_state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            checkpoint_prefix.with_suffix(".pt"),
        )

    def _trainable_state_dict(self):
        state = {}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                state[name] = parameter.detach().cpu()
        return state

    def _average_metrics(self, metrics: dict[str, float], steps: int) -> dict[str, float]:
        divisor = max(1, steps)
        return {name: float(value / divisor) for name, value in metrics.items()}

    def _log_model_summary(self) -> None:
        model_total, model_trainable = count_parameters(self.model)
        target_total, target_trainable = count_parameters(self.target_model)
        print(f"[device] using {self.device}")
        print(f"[config] setting={self.setting_name}")
        print(f"[data] loader={self.config.data_loader} source={self.config.data} path={self.config.data_path}")
        print(
            "[model] "
            f"second_workpoint_total={model_total} "
            f"second_workpoint_trainable={model_trainable} "
            f"target_total={target_total} "
            f"target_trainable={target_trainable}"
        )
        print(
            "[freeze] "
            f"backbone={self.config.backbone_mode}:{self.config.backbone_model_name or self.config.backbone_model_path} "
            f"target_model={self.config.target_model_mode}:{self.config.target_model}"
        )
        print(
            f"[semantic] mode={self.config.semantic_alignment_mode} "
            f"prompt_max_tokens={self.config.backbone_prompt_max_tokens}"
        )
        print("[trainable] temporal_adapter visual_adapter shared_reprogramming projector output_head")
