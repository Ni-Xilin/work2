"""训练器模块。
这个文件负责把配置、数据、主模型、目标反馈模型和损失函数串起来，
形成一个最小但完整的训练闭环，并保存 checkpoint 与训练摘要。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from second_workpoint.config import ExperimentConfig
from second_workpoint.data.factory import build_dataset
from second_workpoint.models.second_workpoint_model import SecondWorkpointStubModel
from second_workpoint.models.target_model_adapter import build_target_model
from second_workpoint.training.losses import TargetedOverheadLoss
from second_workpoint.utils.runtime import (
    count_parameters,
    ensure_dir,
    import_torch,
    resolve_device,
    resolve_torch_device,
    save_json,
    save_npz,
    sigmoid,
)


class StubTrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = resolve_device(config)
        self.setting_name = config.setting_name()
        self.run_dir = ensure_dir(Path(config.checkpoints) / self.setting_name)

        self.train_dataset = build_dataset(config, split="train")
        self.eval_dataset = build_dataset(config, split="eval")

        self.model = SecondWorkpointStubModel(config)
        self.target_model = build_target_model(config)
        self.criterion = TargetedOverheadLoss(config)

        self._log_model_summary()

    def train(self) -> dict:
        history: list[dict] = []

        for epoch in range(1, self.config.train_epochs + 1):
            epoch_metrics = {
                "loss": 0.0,
                "label_loss": 0.0,
                "time_ratio": 0.0,
                "size_ratio": 0.0,
                "update_norm": 0.0,
            }

            for step, batch in enumerate(self._iterate_batches(self.train_dataset, shuffle=True, epoch=epoch), start=1):
                step_outputs = self._forward_batch(batch, log_shapes=(epoch == 1 and step == 1))
                grad_perturbation = self._build_stub_gradient(step_outputs)
                update_norm = float(np.linalg.norm(grad_perturbation.reshape(grad_perturbation.shape[0], -1), axis=1).mean())
                self.model.apply_stub_update(
                    outputs=step_outputs["model_outputs"],
                    grad_perturbation=grad_perturbation,
                    learning_rate=self.config.learning_rate,
                    update_scale=self.config.stub_update_scale,
                )

                epoch_metrics["loss"] += step_outputs["loss"]
                epoch_metrics["label_loss"] += step_outputs["label_loss"]
                epoch_metrics["time_ratio"] += step_outputs["time_ratio"]
                epoch_metrics["size_ratio"] += step_outputs["size_ratio"]
                epoch_metrics["update_norm"] += update_norm

                if step % self.config.log_interval == 0:
                    print(
                        f"[train] epoch={epoch} step={step} "
                        f"loss={step_outputs['loss']:.4f} "
                        f"label={step_outputs['label_loss']:.4f} "
                        f"time={step_outputs['time_ratio']:.4f} "
                        f"size={step_outputs['size_ratio']:.4f} "
                        f"update_norm={update_norm:.4f}"
                    )

            train_summary = self._average_metrics(epoch_metrics, self._num_batches(self.train_dataset))
            eval_summary = self.evaluate()
            history.append(
                {
                    "epoch": epoch,
                    "train": train_summary,
                    "eval": eval_summary,
                }
            )
            print(
                f"[epoch] {epoch} "
                f"train_loss={train_summary['loss']:.4f} "
                f"eval_loss={eval_summary['loss']:.4f}"
            )
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
        metrics = {
            "loss": 0.0,
            "label_loss": 0.0,
            "time_ratio": 0.0,
            "size_ratio": 0.0,
        }

        for batch in self._iterate_batches(self.eval_dataset, shuffle=False, epoch=0):
            outputs = self._forward_batch(batch, log_shapes=False)
            metrics["loss"] += outputs["loss"]
            metrics["label_loss"] += outputs["label_loss"]
            metrics["time_ratio"] += outputs["time_ratio"]
            metrics["size_ratio"] += outputs["size_ratio"]

        return self._average_metrics(metrics, self._num_batches(self.eval_dataset))

    def _iterate_batches(self, dataset, shuffle: bool, epoch: int):
        indices = np.arange(len(dataset))
        rng = np.random.default_rng(self.config.random_seed + epoch + (17 if shuffle else 1017))
        if shuffle:
            rng.shuffle(indices)

        for start in range(0, len(indices), self.config.batch_size):
            batch_indices = indices[start : start + self.config.batch_size]
            samples = [dataset[int(index)] for index in batch_indices]
            yield self._stack_batch(samples)

    def _stack_batch(self, samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray | list[str]]:
        batch: dict[str, np.ndarray | list[str]] = {}
        for key in samples[0]:
            first_value = samples[0][key]
            if isinstance(first_value, str):
                batch[key] = [str(sample[key]) for sample in samples]
                continue
            batch[key] = np.stack([sample[key] for sample in samples], axis=0)
        return batch

    def _forward_batch(self, batch: dict[str, np.ndarray], log_shapes: bool) -> dict[str, object]:
        full_flow = batch["full_flow"].astype(np.float32)
        history_seq = batch["history_seq"].astype(np.float32)
        clean_future = batch["clean_future"].astype(np.float32)
        prompt_ids = batch["prompt_ids"].astype(np.int64)
        future_mask = (
            batch["future_mask"].astype(np.float32)
            if "future_mask" in batch
            else np.ones((full_flow.shape[0], self.config.pred_len), dtype=np.float32)
        )
        writeback_meta = (
            batch["writeback_meta"].astype(np.int64)
            if "writeback_meta" in batch
            else np.asarray(
                [[self.config.seq_len, self.config.pred_len] for _ in range(full_flow.shape[0])],
                dtype=np.int64,
            )
        )
        future_mask_3d = future_mask[:, :, None]

        model_outputs = self.model.forward(history_seq=history_seq, prompt_ids=prompt_ids)
        perturbation = model_outputs["perturbation"].astype(np.float32)
        effective_perturbation = perturbation * future_mask_3d
        positive_delta = np.abs(effective_perturbation)
        direction = np.where(clean_future >= 0.0, 1.0, -1.0).astype(np.float32)
        adv_future = direction * (np.abs(clean_future) + positive_delta)

        adv_flow = full_flow.copy()
        flow_mask = np.ones((full_flow.shape[0], full_flow.shape[2]), dtype=np.float32)
        target_original_flow = (
            batch["target_full_flow"].astype(np.float32).copy()
            if "target_full_flow" in batch
            else full_flow.copy()
        )
        target_adv_flow = target_original_flow.copy()

        for sample_index in range(full_flow.shape[0]):
            writeback_start = int(writeback_meta[sample_index, 0])
            valid_length = int(writeback_meta[sample_index, 1])
            flow_mask[sample_index, writeback_start : writeback_start + self.config.pred_len] = future_mask[sample_index]
            if valid_length <= 0:
                continue
            writeback_end = writeback_start + valid_length
            adv_flow[sample_index, :, writeback_start:writeback_end] = adv_future[sample_index, :valid_length, :].T
            self._write_target_future(
                target_adv_flow=target_adv_flow,
                sample_index=sample_index,
                writeback_start=writeback_start,
                valid_length=valid_length,
                adv_future=adv_future,
            )

        target_logits = self.target_model.forward(target_adv_flow)
        target_labels = np.zeros_like(target_logits, dtype=np.float32)
        loss_outputs = self.criterion.forward(
            target_logits=target_logits,
            target_labels=target_labels,
            original_flow=full_flow,
            adv_flow=adv_flow,
            flow_mask=flow_mask,
        )

        if log_shapes:
            x_ts_shape = batch["x_ts"].shape if "x_ts" in batch else None
            x_vis_shape = batch["x_vis"].shape if "x_vis" in batch else None
            print(
                "[shape] "
                f"history_seq={history_seq.shape} "
                f"x_ts={x_ts_shape} "
                f"x_vis={x_vis_shape} "
                f"temporal_tokens={model_outputs['temporal_tokens'].shape} "
                f"visual_tokens={model_outputs['visual_tokens'].shape} "
                f"context_tokens={model_outputs['context_tokens'].shape} "
                f"perturbation={perturbation.shape} "
                f"future_mask={future_mask.shape} "
                f"adv_flow={adv_flow.shape} "
                f"target_adv_flow={target_adv_flow.shape} "
                f"logits={target_logits.shape}"
            )

        return {
            "loss": float(loss_outputs["loss"]),
            "label_loss": float(loss_outputs["label_loss"]),
            "time_ratio": float(loss_outputs["time_ratio"]),
            "size_ratio": float(loss_outputs["size_ratio"]),
            "target_logits": target_logits,
            "adv_flow": adv_flow,
            "target_adv_flow": target_adv_flow,
            "clean_future": clean_future,
            "effective_perturbation": effective_perturbation,
            "future_mask": future_mask,
            "model_outputs": model_outputs,
        }

    def _build_stub_gradient(self, step_outputs: dict[str, object]) -> np.ndarray:
        perturbation = step_outputs["effective_perturbation"].astype(np.float32)
        target_logits = step_outputs["target_logits"].astype(np.float32)
        target_pressure = float(sigmoid(target_logits).mean())

        gradient = np.sign(perturbation).astype(np.float32)
        gradient *= self.config.beta * target_pressure

        if self.config.time_channel_indices:
            time_channels = perturbation[:, :, self.config.time_channel_indices]
            time_scale = self.config.alpha * float(np.mean(np.abs(time_channels)))
            gradient[:, :, self.config.time_channel_indices] += time_scale * np.sign(time_channels)

        if self.config.size_channel_indices:
            size_channels = perturbation[:, :, self.config.size_channel_indices]
            size_scale = self.config.gamma * float(np.mean(np.abs(size_channels)))
            gradient[:, :, self.config.size_channel_indices] += size_scale * np.sign(size_channels)

        return gradient.astype(np.float32)

    def _write_target_future(
        self,
        target_adv_flow: np.ndarray,
        sample_index: int,
        writeback_start: int,
        valid_length: int,
        adv_future: np.ndarray,
    ) -> None:
        writeback_end = writeback_start + valid_length
        future_slice = adv_future[sample_index, :valid_length, :].T
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
        save_npz(
            checkpoint_prefix.with_suffix(".npz"),
            self.model.state_dict(),
        )

    def _average_metrics(self, metrics: dict[str, float], num_steps: int) -> dict[str, float]:
        divisor = max(1, num_steps)
        return {name: float(value / divisor) for name, value in metrics.items()}

    def _num_batches(self, dataset) -> int:
        return max(1, (len(dataset) + self.config.batch_size - 1) // self.config.batch_size)

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
            f"[freeze] backbone=FrozenBackboneStub target_model={self.config.target_model_mode}:{self.config.target_model}"
        )
        print(
            f"[semantic] mode={self.config.semantic_alignment_mode} "
            f"prompt_max_tokens={self.config.backbone_prompt_max_tokens}"
        )
        print("[trainable] temporal_adapter visual_adapter shared_reprogramming output_head")


def build_trainer(config: ExperimentConfig):
    if config.backend == "numpy_stub":
        return StubTrainer(config)
    if config.backend == "torch_real":
        return TorchTrainer(config)
    raise ValueError(f"Unsupported backend: {config.backend}")


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
