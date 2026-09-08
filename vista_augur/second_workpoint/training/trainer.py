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
from second_workpoint.training.deepcoffea_protocol import aggregate_session_scores, partition_sessions_by_ipd
from second_workpoint.training.losses import TargetedOverheadLoss
from second_workpoint.training.metrics import summarize_scores, threshold_at_target_fpr
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

        self.train_dataset = build_dataset(config, split="train") if config.is_training else None
        skip_epoch_eval_dataset = config.is_training and "deepcoffea" in config.target_model.lower()
        self.eval_dataset = None if skip_epoch_eval_dataset else build_dataset(config, split="eval")

        from torch.utils.data import DataLoader
        from second_workpoint.models import SecondWorkpointTorchRealModel

        self.train_loader = None
        if self.train_dataset is not None:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                collate_fn=self._torch_collate_batch,
                drop_last=config.training_drop_last,
            )
        self.eval_loader = None
        if self.eval_dataset is not None:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=self._torch_collate_batch,
                drop_last=bool(config.is_training and config.training_drop_last),
            )

        self.model = SecondWorkpointTorchRealModel(config)
        self.target_model = build_target_model(config)
        self.criterion = TargetedOverheadLoss(config)
        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = self.torch.optim.Adam(trainable_parameters, lr=config.learning_rate)
        self.scheduler = self.torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=float(config.learning_rate_decay),
        )
        self.start_epoch = 1
        self.best_eval_loss = float("inf")
        self.epochs_without_improvement = 0
        if config.resume_from_checkpoint:
            self._load_checkpoint(Path(config.resume_from_checkpoint))

        self._log_model_summary()

    def train(self) -> dict:
        if self.train_loader is None:
            raise RuntimeError("Training was requested without a training dataloader.")
        history: list[dict] = []
        for epoch in range(self.start_epoch, self.config.train_epochs + 1):
            self.model.train()
            epoch_metrics = self._empty_metric_totals()
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
                    accumulated_steps = ((step - 1) % self.config.gradient_accumulation_steps) + 1
                    if (
                        "deepcoffea" in self.config.target_model.lower()
                        and accumulated_steps < self.config.gradient_accumulation_steps
                    ):
                        correction = float(self.config.gradient_accumulation_steps) / float(accumulated_steps)
                        for parameter in self.model.parameters():
                            if parameter.requires_grad and parameter.grad is not None:
                                parameter.grad.mul_(correction)
                    if self.config.max_grad_norm > 0:
                        self.torch.nn.utils.clip_grad_norm_(
                            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                            max_norm=float(self.config.max_grad_norm),
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                self._accumulate_metrics(epoch_metrics, outputs)
                steps += 1

                if step % self.config.log_interval == 0:
                    if self._uses_work1_deepcoffea_training_monitor():
                        print(
                            f"[train] epoch={epoch} step={step} "
                            f"loss={float(outputs['loss'].detach().item()):.4f} "
                            f"cosine_loss={outputs['label_loss']:.4f} "
                            f"time={outputs['time_ratio']:.4f} "
                            f"size={outputs['size_ratio']:.4f} "
                            f"adv_similarity={outputs['mean_adv_logit']:.4f}"
                        )
                    else:
                        print(
                            f"[train] epoch={epoch} step={step} "
                            f"loss={float(outputs['loss'].detach().item()):.4f} "
                            f"label={outputs['label_loss']:.4f} "
                            f"time={outputs['time_ratio']:.4f} "
                            f"size={outputs['size_ratio']:.4f} "
                            f"orig_positive={outputs['original_positive_rate']:.4f} "
                            f"adv_positive={outputs['adv_positive_rate']:.4f} "
                            f"positive_drop={outputs['positive_rate_drop']:.4f} "
                            f"asr={outputs['attack_success_rate']:.4f} "
                            f"flip={outputs['flip_rate']:.4f} "
                            f"orig_logit={outputs['mean_original_logit']:.4f} "
                            f"adv_logit={outputs['mean_adv_logit']:.4f} "
                            f"orig_prob={outputs['mean_original_prob']:.4f} "
                            f"adv_prob={outputs['mean_adv_prob']:.4f}"
                        )
                if self.config.max_train_steps > 0 and steps >= self.config.max_train_steps:
                    break

            train_summary = self._finalize_metrics(epoch_metrics)
            work1_deepcoffea_training = self._uses_work1_deepcoffea_training_monitor()
            if work1_deepcoffea_training:
                # Work1 does not run its test loader after a DeepCoFFEA training
                # epoch. It monitors the mean training cosine hinge and L2 rates.
                eval_summary = None
                monitor_summary = train_summary
            else:
                # DeepCorr/mDeepCorr retain their completed epoch-validation path.
                eval_summary = self.evaluate(include_negatives=False)
                monitor_summary = eval_summary
            history.append({"epoch": epoch, "train": train_summary, "eval": eval_summary})
            if work1_deepcoffea_training:
                print(
                    f"[epoch] {epoch} "
                    f"train_loss={train_summary['loss']:.4f} "
                    f"cosine_loss={train_summary['label_loss']:.4f} "
                    f"time={train_summary['time_ratio']:.4f} "
                    f"size={train_summary['size_ratio']:.4f} "
                    f"adv_similarity={train_summary['mean_adv_logit']:.4f}"
                )
            else:
                print(
                    f"[epoch] {epoch} "
                    f"train_loss={train_summary['loss']:.4f} "
                    f"eval_loss={eval_summary['loss']:.4f} "
                    f"train_orig_positive={train_summary['original_positive_rate']:.4f} "
                    f"train_adv_positive={train_summary['adv_positive_rate']:.4f} "
                    f"eval_orig_positive={eval_summary['original_positive_rate']:.4f} "
                    f"eval_adv_positive={eval_summary['adv_positive_rate']:.4f} "
                    f"eval_positive_drop={eval_summary['positive_rate_drop']:.4f} "
                    f"train_asr={train_summary['attack_success_rate']:.4f} "
                    f"eval_asr={eval_summary['attack_success_rate']:.4f} "
                    f"train_flip={train_summary['flip_rate']:.4f} "
                    f"eval_flip={eval_summary['flip_rate']:.4f} "
                    f"eval_clean_f1={eval_summary['clean_f1']:.4f} "
                    f"eval_adv_f1={eval_summary['adv_f1']:.4f} "
                    f"train_adv_prob={train_summary['mean_adv_prob']:.4f} "
                    f"eval_adv_prob={eval_summary['mean_adv_prob']:.4f}"
                )
            improved = monitor_summary["loss"] < self.best_eval_loss
            if improved:
                self.best_eval_loss = monitor_summary["loss"]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            self._step_learning_rate(epoch)
            self._save_checkpoint(
                epoch,
                train_summary,
                eval_summary,
                is_best=improved,
                monitor_summary=monitor_summary,
            )
            if (
                not work1_deepcoffea_training
                and self.config.patience > 0
                and self.epochs_without_improvement >= self.config.patience
            ):
                print(f"[early-stop] no eval loss improvement for {self.config.patience} epochs")
                break

        summary = {
            "setting_name": self.setting_name,
            "device": self.device,
            "backend": self.config.backend,
            "epochs": history,
        }
        save_json(self.run_dir / "train_summary.json", summary)
        return summary

    def _uses_work1_deepcoffea_training_monitor(self) -> bool:
        return "deepcoffea" in self.config.target_model.lower()

    def _step_learning_rate(self, completed_epoch: int) -> None:
        if "deepcoffea" in self.config.target_model.lower() and self.config.lradj == "type4":
            exponent = max(0, int(completed_epoch) - 2)
            next_lr = float(self.config.learning_rate) * (0.7**exponent)
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = next_lr
            return
        self.scheduler.step()

    def evaluate(self, include_negatives: bool = True) -> dict:
        if self.eval_loader is None:
            raise RuntimeError("Evaluation loader is unavailable during Work1-aligned DeepCoFFEA training.")
        self.model.eval()
        metrics = self._empty_metric_totals()
        score_buffers = {
            "clean_positive_scores": [],
            "clean_negative_scores": [],
            "adv_positive_scores": [],
            "adv_negative_scores": [],
        }
        steps = 0
        with self.torch.no_grad():
            for step, batch in enumerate(self.eval_loader, start=1):
                outputs = self._forward_batch(batch, log_shapes=False, include_negatives=include_negatives)
                self._accumulate_metrics(metrics, outputs)
                for name in score_buffers:
                    score_buffers[name].extend(outputs.get(name, []))
                steps += 1
                if self.config.max_eval_steps > 0 and steps >= self.config.max_eval_steps:
                    break
        summary = self._finalize_metrics(metrics)
        summary["operating_points"] = self._build_operating_points(score_buffers)
        if include_negatives and score_buffers["clean_negative_scores"]:
            score_path = self.run_dir / "evaluation_scores.npz"
            np.savez_compressed(
                score_path,
                **{name: np.asarray(values, dtype=np.float32) for name, values in score_buffers.items()},
            )
            summary["score_artifact"] = score_path.name
        return summary

    def _torch_collate_batch(self, samples):
        if samples[0].get("dataset_protocol") == "deepcoffea_session":
            return self._collate_deepcoffea_sessions(samples)
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

    def _collate_deepcoffea_sessions(self, samples):
        variable_tensor_keys = {
            "full_flow",
            "history_seq",
            "clean_future",
            "future_mask",
            "writeback_meta",
        }
        batch = {"dataset_protocol": "deepcoffea_session"}
        for key in samples[0]:
            if key == "dataset_protocol":
                continue
            values = [sample[key] for sample in samples]
            if key in variable_tensor_keys:
                batch[key] = [self.torch.from_numpy(value).float() for value in values]
            elif key == "prompt_text":
                batch[key] = values
            elif isinstance(values[0], str):
                batch[key] = [str(value) for value in values]
            else:
                stacked = np.stack(values, axis=0)
                batch[key] = self.torch.from_numpy(stacked).float()
        return batch

    def _forward_deepcoffea_batch(self, batch, log_shapes: bool, include_negatives: bool = False):
        sessions = [flow.to(self.trainable_device, dtype=self.torch.float32) for flow in batch["full_flow"]]
        histories = [value.to(self.trainable_device, dtype=self.torch.float32) for value in batch["history_seq"]]
        futures = [value.to(self.trainable_device, dtype=self.torch.float32) for value in batch["clean_future"]]
        future_masks = [value.to(self.trainable_device, dtype=self.torch.float32) for value in batch["future_mask"]]
        writeback = [value.to(self.trainable_device, dtype=self.torch.long) for value in batch["writeback_meta"]]
        window_counts = [int(value.shape[0]) for value in histories]
        history_seq = self.torch.cat(histories, dim=0)
        clean_future = self.torch.cat(futures, dim=0)
        future_mask = self.torch.cat(future_masks, dim=0)
        writeback_meta = self.torch.cat(writeback, dim=0)
        prompt_text = [text for session_prompts in batch["prompt_text"] for text in session_prompts]
        window_owner = self.torch.cat(
            [
                self.torch.full((count,), owner, device=self.trainable_device, dtype=self.torch.long)
                for owner, count in enumerate(window_counts)
            ]
        )

        perturbation = self._generate_deepcoffea_perturbations(history_seq, prompt_text)
        positive_delta = perturbation.abs() * future_mask.unsqueeze(-1)
        channel_mask = self.torch.ones((self.config.enc_in,), device=self.trainable_device)
        if self.config.adv_type == "time":
            channel_mask[self.config.size_channel_indices] = 0.0
        elif self.config.adv_type == "size":
            channel_mask[self.config.time_channel_indices] = 0.0
        positive_delta = positive_delta * channel_mask.view(1, 1, -1)
        adv_future = self.torch.sign(clean_future) * (clean_future.abs() + positive_delta)

        adv_sessions = [session.clone() for session in sessions]
        for future_index in range(history_seq.shape[0]):
            owner = int(window_owner[future_index].item())
            start = int(writeback_meta[future_index, 0].item())
            valid_length = int(writeback_meta[future_index, 1].item())
            if valid_length > 0:
                adv_sessions[owner][:, start : start + valid_length] = adv_future[
                    future_index, :valid_length
                ].transpose(0, 1)

        original_flow, flow_mask = self._pad_deepcoffea_sessions(sessions)
        adv_flow, _ = self._pad_deepcoffea_sessions(adv_sessions)
        exit_windows = batch["target_exit_windows"].to(self.trainable_device, dtype=self.torch.float32)

        adv_tor_windows = partition_sessions_by_ipd(
            adv_sessions,
            delta_seconds=self.config.deepcoffea_delta_seconds,
            window_seconds=self.config.deepcoffea_window_seconds,
            window_count=self.config.deepcoffea_n_windows,
            packet_limit=self.config.deepcoffea_tor_len,
            mask_steepness=self.config.deepcoffea_partition_steepness,
        )
        adv_window_logits = self.target_model.forward(adv_tor_windows, exit_flow=exit_windows).to(self.trainable_device)
        target_labels = self.torch.zeros_like(adv_window_logits)
        loss_outputs = self.criterion.forward(
            target_logits=adv_window_logits,
            target_labels=target_labels,
            original_flow=original_flow,
            adv_flow=adv_flow,
            flow_mask=flow_mask,
        )

        batch_size = len(sessions)
        if log_shapes:
            print(
                "[shape] deepcoffea_session "
                f"sessions={batch_size} generator_windows={tuple(history_seq.shape)} "
                f"target_windows={tuple(adv_tor_windows.shape)} logits={tuple(adv_window_logits.shape)}"
            )
        if self.config.is_training and not include_negatives:
            return {
                "loss": loss_outputs["loss"],
                "label_loss": float(loss_outputs["label_loss"].detach().item()),
                "time_ratio": float(loss_outputs["time_ratio"].detach().item()),
                "size_ratio": float(loss_outputs["size_ratio"].detach().item()),
                "batch_weight": batch_size,
                "mean_adv_logit": float(adv_window_logits.detach().mean().item()),
            }

        clean_tor_windows = batch["target_tor_windows"].to(self.trainable_device, dtype=self.torch.float32)
        with self.torch.no_grad():
            original_window_logits = self.target_model.forward(clean_tor_windows, exit_flow=exit_windows).to(
                self.trainable_device
            )
        original_session_scores = aggregate_session_scores(
            original_window_logits.reshape(batch_size, self.config.deepcoffea_n_windows),
            self.config.deepcoffea_vote_threshold,
        )
        adv_session_scores = aggregate_session_scores(
            adv_window_logits.reshape(batch_size, self.config.deepcoffea_n_windows),
            self.config.deepcoffea_vote_threshold,
        )
        metric_outputs = self._compute_attack_metrics(original_session_scores, adv_session_scores)
        metric_outputs["mean_original_logit"] = float(original_window_logits.detach().mean().item())
        metric_outputs["mean_adv_logit"] = float(adv_window_logits.detach().mean().item())
        metric_outputs["mean_original_prob"] = metric_outputs["mean_original_logit"]
        metric_outputs["mean_adv_prob"] = metric_outputs["mean_adv_logit"]
        classification_outputs = self._compute_deepcoffea_classification_counts(
            clean_tor_windows=clean_tor_windows,
            adv_tor_windows=adv_tor_windows,
            exit_windows=exit_windows,
            negative_exit_windows=batch.get("target_negative_exit_windows"),
            positive_original_scores=original_session_scores,
            positive_adv_scores=adv_session_scores,
            include_negatives=include_negatives,
        )

        return {
            "loss": loss_outputs["loss"],
            "label_loss": float(loss_outputs["label_loss"].detach().item()),
            "time_ratio": float(loss_outputs["time_ratio"].detach().item()),
            "size_ratio": float(loss_outputs["size_ratio"].detach().item()),
            **metric_outputs,
            **classification_outputs,
        }

    def _generate_deepcoffea_perturbations(self, history_seq, prompt_text):
        chunk_size = int(self.config.deepcoffea_generator_window_batch_size)
        if chunk_size <= 0 or history_seq.shape[0] <= chunk_size:
            return self.model.forward(history_seq=history_seq, prompt_text=prompt_text)["perturbation"]
        chunks = []
        for start in range(0, history_seq.shape[0], chunk_size):
            end = min(start + chunk_size, history_seq.shape[0])
            chunks.append(
                self.model.forward(
                    history_seq=history_seq[start:end],
                    prompt_text=prompt_text[start:end],
                )["perturbation"]
            )
        return self.torch.cat(chunks, dim=0)

    def _pad_deepcoffea_sessions(self, sessions):
        max_length = max(int(session.shape[1]) for session in sessions)
        padded = []
        masks = []
        for session in sessions:
            padding = max_length - int(session.shape[1])
            padded.append(self.torch.nn.functional.pad(session, (0, padding)))
            masks.append(
                self.torch.nn.functional.pad(
                    self.torch.ones(session.shape[1], device=session.device, dtype=session.dtype),
                    (0, padding),
                )
            )
        return self.torch.stack(padded, dim=0), self.torch.stack(masks, dim=0)

    def _forward_batch(self, batch, log_shapes: bool, include_negatives: bool = False):
        if batch.get("dataset_protocol") == "deepcoffea_session":
            return self._forward_deepcoffea_batch(batch, log_shapes, include_negatives)
        return self._forward_standard_batch(batch, log_shapes, include_negatives)

    def _forward_standard_batch(self, batch, log_shapes: bool, include_negatives: bool = False):
        full_flow = batch["full_flow"].to(self.trainable_device, dtype=self.torch.float32)
        history_seq = batch["history_seq"].to(self.trainable_device, dtype=self.torch.float32)
        clean_future = batch["clean_future"].to(self.trainable_device, dtype=self.torch.float32)
        grouped_windows = history_seq.ndim == 4
        if grouped_windows:
            batch_size, window_count, channel_count, history_length = history_seq.shape
            history_seq = history_seq.reshape(batch_size * window_count, channel_count, history_length)
            clean_future = clean_future.reshape(batch_size * window_count, self.config.pred_len, self.config.enc_in)
            prompt_text = [text for sample_prompts in batch["prompt_text"] for text in sample_prompts]
            window_owner = self.torch.arange(batch_size, device=self.trainable_device).repeat_interleave(window_count)
        else:
            prompt_text = list(batch["prompt_text"])
            window_owner = self.torch.arange(full_flow.shape[0], device=self.trainable_device)
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
        if grouped_windows:
            future_mask = future_mask.reshape(batch_size * window_count, self.config.pred_len)
            writeback_meta = writeback_meta.reshape(batch_size * window_count, 2)
        future_mask_3d = future_mask.unsqueeze(-1)

        model_outputs = self.model.forward(history_seq=history_seq, prompt_text=prompt_text)
        perturbation = model_outputs["perturbation"]
        effective_perturbation = perturbation * future_mask_3d
        positive_delta = effective_perturbation.abs()
        channel_mask = self.torch.ones((self.config.enc_in,), device=self.trainable_device)
        if self.config.adv_type == "time":
            channel_mask[self.config.size_channel_indices] = 0.0
        elif self.config.adv_type == "size":
            channel_mask[self.config.time_channel_indices] = 0.0
        positive_delta = positive_delta * channel_mask.view(1, 1, -1)
        # The sign encodes packet direction only; the generated value contributes
        # exclusively to the non-negative time/size magnitude.  torch.sign also
        # keeps padded or otherwise empty zero positions at zero.
        direction = self.torch.sign(clean_future)
        adv_future = direction * (clean_future.abs() + positive_delta)

        adv_flow = full_flow.clone()
        flow_mask = self.torch.ones((full_flow.shape[0], full_flow.shape[2]), dtype=self.torch.float32, device=self.trainable_device)
        target_original_flow = (
            batch["target_full_flow"].to(self.trainable_device, dtype=self.torch.float32).clone()
            if "target_full_flow" in batch
            else full_flow.clone()
        )
        target_adv_flow = target_original_flow.clone()
        target_negative_flow = (
            batch["target_negative_flow"].to(self.trainable_device, dtype=self.torch.float32)
            if "target_negative_flow" in batch
            else None
        )
        target_negative_adv_flow = None
        target_exit_flow = (
            batch["target_exit_flow"].to(self.trainable_device, dtype=self.torch.float32)
            if "target_exit_flow" in batch
            else None
        )
        target_negative_exit_flow = (
            batch["target_negative_exit_flow"].to(self.trainable_device, dtype=self.torch.float32)
            if "target_negative_exit_flow" in batch
            else None
        )
        with self.torch.no_grad():
            original_target_logits = self.target_model.forward(
                target_original_flow,
                exit_flow=target_exit_flow,
            ).to(self.trainable_device)

        for window_index in range(history_seq.shape[0]):
            sample_index = int(window_owner[window_index].item())
            writeback_start = int(writeback_meta[window_index, 0].item())
            valid_length = int(writeback_meta[window_index, 1].item())
            flow_mask[sample_index, writeback_start : writeback_start + self.config.pred_len] = future_mask[window_index]
            if valid_length <= 0:
                continue
            writeback_end = writeback_start + valid_length
            adv_flow[sample_index, :, writeback_start:writeback_end] = adv_future[window_index, :valid_length, :].transpose(0, 1)
            self._write_target_future_torch(
                target_adv_flow=target_adv_flow,
                sample_index=sample_index,
                writeback_start=writeback_start,
                valid_length=valid_length,
                adv_future=adv_future,
                future_index=window_index,
            )

        if target_negative_flow is not None:
            target_negative_adv_flow = target_negative_flow.clone()
            for target_row in self.config.tor_row_indices:
                target_negative_adv_flow[:, :, int(target_row), :] = target_adv_flow[:, None, int(target_row), :]

        target_logits = self.target_model.forward(target_adv_flow, exit_flow=target_exit_flow).to(self.trainable_device)
        target_labels = self.torch.zeros_like(target_logits, dtype=self.torch.float32, device=self.trainable_device)
        loss_outputs = self.criterion.forward(
            target_logits=target_logits,
            target_labels=target_labels,
            original_flow=full_flow,
            adv_flow=adv_flow,
            flow_mask=flow_mask,
        )
        metric_outputs = self._compute_attack_metrics(
            original_target_logits=original_target_logits,
            adv_target_logits=target_logits,
        )
        classification_outputs = self._compute_classification_counts(
            target_original_flow=target_original_flow,
            target_adv_flow=target_adv_flow,
            target_exit_flow=target_exit_flow,
            target_negative_flow=target_negative_flow,
            target_negative_adv_flow=target_negative_adv_flow,
            target_negative_exit_flow=target_negative_exit_flow,
            positive_original_logits=original_target_logits,
            positive_adv_logits=target_logits,
            include_negatives=include_negatives,
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
            "batch_weight": metric_outputs["batch_weight"],
            "original_positive_count": metric_outputs["original_positive_count"],
            "adv_positive_count": metric_outputs["adv_positive_count"],
            "successful_count": metric_outputs["successful_count"],
            "flip_count": metric_outputs["flip_count"],
            **classification_outputs,
            "original_positive_rate": metric_outputs["original_positive_rate"],
            "adv_positive_rate": metric_outputs["adv_positive_rate"],
            "positive_rate_drop": metric_outputs["positive_rate_drop"],
            "attack_success_rate": metric_outputs["attack_success_rate"],
            "flip_rate": metric_outputs["flip_rate"],
            "mean_original_logit": metric_outputs["mean_original_logit"],
            "mean_adv_logit": metric_outputs["mean_adv_logit"],
            "mean_original_prob": metric_outputs["mean_original_prob"],
            "mean_adv_prob": metric_outputs["mean_adv_prob"],
        }

    def _write_target_future_torch(
        self,
        target_adv_flow,
        sample_index: int,
        writeback_start: int,
        valid_length: int,
        adv_future,
        future_index: int | None = None,
    ) -> None:
        writeback_end = writeback_start + valid_length
        source_index = sample_index if future_index is None else future_index
        future_slice = adv_future[source_index, :valid_length, :].transpose(0, 1)
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

    def _save_checkpoint(
        self,
        epoch: int,
        train_summary: dict,
        eval_summary: dict | None,
        is_best: bool,
        monitor_summary: dict | None = None,
    ) -> None:
        monitor = monitor_summary or eval_summary
        if monitor is None:
            raise ValueError("Checkpoint saving requires a train or evaluation monitor summary.")
        if self._uses_work1_deepcoffea_training_monitor() and eval_summary is None:
            checkpoint_stem = self._deepcoffea_checkpoint_stem(epoch, monitor)
            monitor_source = "train"
        else:
            checkpoint_stem = self._checkpoint_stem(epoch, monitor)
            monitor_source = "eval"
        checkpoint_prefix = self.run_dir / checkpoint_stem
        save_json(
            checkpoint_prefix.with_suffix(".json"),
            {
                "epoch": epoch,
                "config": self.config.to_dict(),
                "train_summary": train_summary,
                "eval_summary": eval_summary,
                "monitor_source": monitor_source,
                "monitor_summary": monitor,
            },
        )
        payload = {
            "epoch": epoch,
            "config": self.config.to_dict(),
            "train_summary": train_summary,
            "eval_summary": eval_summary,
            "monitor_source": monitor_source,
            "monitor_summary": monitor,
            "best_eval_loss": self.best_eval_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "model_state": self._trainable_state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        self.torch.save(payload, checkpoint_prefix.with_suffix(".pt"))
        self.torch.save(payload, self.run_dir / "latest.pt")
        if is_best:
            self.torch.save(payload, self.run_dir / "best.pt")

    @staticmethod
    def _checkpoint_stem(epoch: int, eval_summary: dict) -> str:
        """生成类似 Work1 但指标语义更准确的 checkpoint 名称。"""

        return (
            f"generator_ep{epoch:03d}"
            f"_origrec{float(eval_summary['original_positive_rate']):.3f}"
            f"_advrec{float(eval_summary['adv_positive_rate']):.3f}"
            f"_loss{float(eval_summary['loss']):.4f}"
            f"_time{float(eval_summary['time_ratio']):.3f}"
            f"_size{float(eval_summary['size_ratio']):.3f}"
        )

    @staticmethod
    def _deepcoffea_checkpoint_stem(epoch: int, train_summary: dict) -> str:
        return (
            f"generator_ep{epoch:03d}"
            f"_cos{float(train_summary['mean_adv_logit']):.3f}"
            f"_loss{float(train_summary['loss']):.4f}"
            f"_time{float(train_summary['time_ratio']):.3f}"
            f"_size{float(train_summary['size_ratio']):.3f}"
        )

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = self.torch.load(checkpoint_path, map_location=self.trainable_device)
        missing, unexpected = self.model.load_state_dict(checkpoint["model_state"], strict=False)
        trainable_names = {name for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        missing_trainable = sorted(name for name in missing if name in trainable_names)
        if missing_trainable or unexpected:
            raise ValueError(
                "Checkpoint is incompatible with the current trainable model: "
                f"missing={missing_trainable} unexpected={sorted(unexpected)}"
            )
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.best_eval_loss = float(checkpoint.get("best_eval_loss", float("inf")))
        self.epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(f"[resume] checkpoint={checkpoint_path} next_epoch={self.start_epoch}")

    def _trainable_state_dict(self):
        state = {}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                state[name] = parameter.detach().cpu()
        return state

    def _empty_metric_totals(self) -> dict[str, float]:
        return {
            "weight": 0.0,
            "loss": 0.0,
            "label_loss": 0.0,
            "time_ratio": 0.0,
            "size_ratio": 0.0,
            "original_positive_count": 0.0,
            "adv_positive_count": 0.0,
            "successful_count": 0.0,
            "flip_count": 0.0,
            "clean_tp": 0.0,
            "clean_fp": 0.0,
            "clean_tn": 0.0,
            "clean_fn": 0.0,
            "adv_tp": 0.0,
            "adv_fp": 0.0,
            "adv_tn": 0.0,
            "adv_fn": 0.0,
            "mean_original_logit": 0.0,
            "mean_adv_logit": 0.0,
            "mean_original_prob": 0.0,
            "mean_adv_prob": 0.0,
        }

    def _accumulate_metrics(self, totals: dict[str, float], outputs: dict) -> None:
        weight = float(outputs["batch_weight"])
        totals["weight"] += weight
        for name in (
            "loss",
            "label_loss",
            "time_ratio",
            "size_ratio",
            "mean_original_logit",
            "mean_adv_logit",
            "mean_original_prob",
            "mean_adv_prob",
        ):
            value = outputs.get(name, 0.0)
            if name == "loss":
                value = value.detach().item()
            totals[name] += float(value) * weight
        for name in (
            "original_positive_count",
            "adv_positive_count",
            "successful_count",
            "flip_count",
            "clean_tp",
            "clean_fp",
            "clean_tn",
            "clean_fn",
            "adv_tp",
            "adv_fp",
            "adv_tn",
            "adv_fn",
        ):
            totals[name] += float(outputs.get(name, 0.0))

    def _finalize_metrics(self, totals: dict[str, float]) -> dict[str, float]:
        weight = max(1.0, totals["weight"])
        original_positive = totals["original_positive_count"]
        summary = {
            name: float(totals[name] / weight)
            for name in (
                "loss",
                "label_loss",
                "time_ratio",
                "size_ratio",
                "mean_original_logit",
                "mean_adv_logit",
                "mean_original_prob",
                "mean_adv_prob",
            )
        }
        summary.update(
            {
                "sample_count": int(totals["weight"]),
                "original_positive_rate": float(original_positive / weight),
                "adv_positive_rate": float(totals["adv_positive_count"] / weight),
                "positive_rate_drop": float((original_positive - totals["adv_positive_count"]) / weight),
                "attack_success_rate": float(totals["successful_count"] / max(1.0, original_positive)),
                "flip_rate": float(totals["flip_count"] / weight),
            }
        )
        summary.update(self._classification_summary(totals, prefix="clean"))
        summary.update(self._classification_summary(totals, prefix="adv"))
        return summary

    def _compute_attack_metrics(self, original_target_logits, adv_target_logits) -> dict[str, float]:
        original_logits = original_target_logits.detach().reshape(-1)
        adv_logits = adv_target_logits.detach().reshape(-1)
        original_scores, threshold = self._decision_scores(original_logits)
        adv_scores, _ = self._decision_scores(adv_logits)
        original_positive = original_scores >= threshold
        adv_positive = adv_scores >= threshold
        flipped = original_positive != adv_positive
        successful = original_positive & (~adv_positive)
        original_positive_count = int(original_positive.sum().item())
        batch_weight = int(original_logits.numel())
        if original_positive_count > 0:
            attack_success_rate = float(successful.float().sum().item() / original_positive_count)
        else:
            attack_success_rate = 0.0
        return {
            "batch_weight": batch_weight,
            "original_positive_count": original_positive_count,
            "adv_positive_count": int(adv_positive.sum().item()),
            "successful_count": int(successful.sum().item()),
            "flip_count": int(flipped.sum().item()),
            "original_positive_rate": float(original_positive.float().mean().item()),
            "adv_positive_rate": float(adv_positive.float().mean().item()),
            "positive_rate_drop": float(original_positive.float().mean().item() - adv_positive.float().mean().item()),
            "attack_success_rate": attack_success_rate,
            "flip_rate": float(flipped.float().mean().item()),
            "mean_original_logit": float(original_logits.mean().item()),
            "mean_adv_logit": float(adv_logits.mean().item()),
            "mean_original_prob": float(original_scores.mean().item()),
            "mean_adv_prob": float(adv_scores.mean().item()),
        }

    def _decision_scores(self, logits):
        if "deepcoffea" in self.config.target_model.lower():
            return logits, float(self.config.deepcoffea_similarity_threshold)
        return self.torch.sigmoid(logits), float(self.config.decision_threshold)

    def _compute_deepcoffea_classification_counts(
        self,
        clean_tor_windows,
        adv_tor_windows,
        exit_windows,
        negative_exit_windows,
        positive_original_scores,
        positive_adv_scores,
        include_negatives: bool,
    ) -> dict[str, int | list[float]]:
        names = ("clean_tp", "clean_fp", "clean_tn", "clean_fn", "adv_tp", "adv_fp", "adv_tn", "adv_fn")
        if not include_negatives:
            return {
                **{name: 0 for name in names},
                "clean_positive_scores": [],
                "clean_negative_scores": [],
                "adv_positive_scores": [],
                "adv_negative_scores": [],
            }
        if negative_exit_windows is None:
            raise ValueError("DeepCoFFEA final evaluation requires deterministic mismatched Exit windows.")

        negative_exit_windows = negative_exit_windows.to(self.trainable_device, dtype=self.torch.float32)
        batch_size, negative_count, window_count, _ = negative_exit_windows.shape
        repeated_clean = clean_tor_windows[:, None].expand(-1, negative_count, -1, -1)
        repeated_adv = adv_tor_windows[:, None].expand(-1, negative_count, -1, -1)
        flat_negative_exit = negative_exit_windows.reshape(-1, negative_exit_windows.shape[-1])
        clean_negative_window_scores = self._target_forward_in_chunks(
            repeated_clean.reshape(-1, repeated_clean.shape[-1]),
            exit_flow=flat_negative_exit,
        ).reshape(batch_size * negative_count, window_count)
        adv_negative_window_scores = self._target_forward_in_chunks(
            repeated_adv.reshape(-1, repeated_adv.shape[-1]),
            exit_flow=flat_negative_exit,
        ).reshape(batch_size * negative_count, window_count)
        clean_negative_scores = aggregate_session_scores(
            clean_negative_window_scores,
            self.config.deepcoffea_vote_threshold,
        )
        adv_negative_scores = aggregate_session_scores(
            adv_negative_window_scores,
            self.config.deepcoffea_vote_threshold,
        )
        threshold = float(self.config.deepcoffea_similarity_threshold)

        def counts(positive_scores, negative_scores, prefix: str) -> dict[str, int]:
            positive_predictions = positive_scores.detach().reshape(-1) >= threshold
            negative_predictions = negative_scores.detach().reshape(-1) >= threshold
            return {
                f"{prefix}_tp": int(positive_predictions.sum().item()),
                f"{prefix}_fn": int((~positive_predictions).sum().item()),
                f"{prefix}_fp": int(negative_predictions.sum().item()),
                f"{prefix}_tn": int((~negative_predictions).sum().item()),
            }

        return {
            **counts(positive_original_scores, clean_negative_scores, "clean"),
            **counts(positive_adv_scores, adv_negative_scores, "adv"),
            "clean_positive_scores": positive_original_scores.detach().float().cpu().tolist(),
            "clean_negative_scores": clean_negative_scores.detach().float().cpu().tolist(),
            "adv_positive_scores": positive_adv_scores.detach().float().cpu().tolist(),
            "adv_negative_scores": adv_negative_scores.detach().float().cpu().tolist(),
        }

    def _compute_classification_counts(
        self,
        target_original_flow,
        target_adv_flow,
        target_exit_flow,
        target_negative_flow,
        target_negative_adv_flow,
        target_negative_exit_flow,
        positive_original_logits,
        positive_adv_logits,
        include_negatives: bool,
    ) -> dict[str, int | list[float]]:
        names = ("clean_tp", "clean_fp", "clean_tn", "clean_fn", "adv_tp", "adv_fp", "adv_tn", "adv_fn")
        if not include_negatives:
            return {
                **{name: 0 for name in names},
                "clean_positive_scores": [],
                "clean_negative_scores": [],
                "adv_positive_scores": [],
                "adv_negative_scores": [],
            }

        if "deepcoffea" in self.config.target_model.lower():
            if target_exit_flow is None:
                raise ValueError("DeepCoFFEA evaluation requires paired exit flows.")
            negative_exit_flow = target_negative_exit_flow
            if negative_exit_flow is None:
                if target_original_flow.shape[0] < 2:
                    return {
                        **{name: 0 for name in names},
                        "clean_positive_scores": [],
                        "clean_negative_scores": [],
                        "adv_positive_scores": [],
                        "adv_negative_scores": [],
                    }
                negative_exit_flow = self.torch.roll(target_exit_flow, shifts=1, dims=0)
                negative_exit_flow = negative_exit_flow.unsqueeze(1)
            negative_count = negative_exit_flow.shape[1]
            repeated_original = target_original_flow[:, None].expand(-1, negative_count, -1, -1)
            repeated_adv = target_adv_flow[:, None].expand(-1, negative_count, -1, -1)
            negative_original_logits = self._target_forward_in_chunks(
                repeated_original.reshape(-1, *target_original_flow.shape[1:]),
                exit_flow=negative_exit_flow.reshape(-1, *negative_exit_flow.shape[2:]),
            )
            negative_adv_logits = self._target_forward_in_chunks(
                repeated_adv.reshape(-1, *target_adv_flow.shape[1:]),
                exit_flow=negative_exit_flow.reshape(-1, *negative_exit_flow.shape[2:]),
            )
        else:
            negative_original_flow = target_negative_flow
            negative_adv_flow = target_negative_adv_flow
            if negative_original_flow is None or negative_adv_flow is None:
                if target_original_flow.shape[0] < 2:
                    return {
                        **{name: 0 for name in names},
                        "clean_positive_scores": [],
                        "clean_negative_scores": [],
                        "adv_positive_scores": [],
                        "adv_negative_scores": [],
                    }
                exit_rows = [index for index in range(target_original_flow.shape[1]) if index not in self.config.tor_row_indices]
                negative_original_flow = target_original_flow.clone()
                negative_adv_flow = target_adv_flow.clone()
                negative_original_flow[:, exit_rows, :] = self.torch.roll(
                    target_original_flow[:, exit_rows, :], shifts=1, dims=0
                )
                negative_adv_flow[:, exit_rows, :] = self.torch.roll(
                    target_adv_flow[:, exit_rows, :], shifts=1, dims=0
                )
                negative_original_flow = negative_original_flow.unsqueeze(1)
                negative_adv_flow = negative_adv_flow.unsqueeze(1)
            negative_original_logits = self._target_forward_in_chunks(
                negative_original_flow.reshape(-1, *negative_original_flow.shape[2:])
            )
            negative_adv_logits = self._target_forward_in_chunks(
                negative_adv_flow.reshape(-1, *negative_adv_flow.shape[2:])
            )

        positive_original_scores, threshold = self._decision_scores(positive_original_logits.detach().reshape(-1))
        positive_adv_scores, _ = self._decision_scores(positive_adv_logits.detach().reshape(-1))
        negative_original_scores, _ = self._decision_scores(negative_original_logits.detach().reshape(-1))
        negative_adv_scores, _ = self._decision_scores(negative_adv_logits.detach().reshape(-1))

        def counts(positive_scores, negative_scores, prefix: str) -> dict[str, int]:
            positive_predictions = positive_scores >= threshold
            negative_predictions = negative_scores >= threshold
            return {
                f"{prefix}_tp": int(positive_predictions.sum().item()),
                f"{prefix}_fn": int((~positive_predictions).sum().item()),
                f"{prefix}_fp": int(negative_predictions.sum().item()),
                f"{prefix}_tn": int((~negative_predictions).sum().item()),
            }

        return {
            **counts(positive_original_scores, negative_original_scores, "clean"),
            **counts(positive_adv_scores, negative_adv_scores, "adv"),
            "clean_positive_scores": positive_original_scores.detach().float().cpu().tolist(),
            "clean_negative_scores": negative_original_scores.detach().float().cpu().tolist(),
            "adv_positive_scores": positive_adv_scores.detach().float().cpu().tolist(),
            "adv_negative_scores": negative_adv_scores.detach().float().cpu().tolist(),
        }

    def _build_operating_points(self, score_buffers: dict[str, list[float]]) -> dict[str, dict]:
        if not score_buffers["clean_negative_scores"]:
            return {}
        results = {}
        for target_fpr in self.config.report_fpr_targets:
            clean_calibration = threshold_at_target_fpr(score_buffers["clean_negative_scores"], target_fpr)
            adv_calibration = threshold_at_target_fpr(score_buffers["adv_negative_scores"], target_fpr)
            key = f"fpr_{target_fpr:.0e}"
            results[key] = {
                "clean": {
                    **clean_calibration,
                    **summarize_scores(
                        score_buffers["clean_positive_scores"],
                        score_buffers["clean_negative_scores"],
                        float(clean_calibration["threshold"]),
                    ),
                },
                "adversarial": {
                    **adv_calibration,
                    **summarize_scores(
                        score_buffers["adv_positive_scores"],
                        score_buffers["adv_negative_scores"],
                        float(adv_calibration["threshold"]),
                    ),
                },
            }
        return results

    def _target_forward_in_chunks(self, flow, exit_flow=None):
        """Evaluate Work1-scale negative pairs without materializing target activations at once."""

        outputs = []
        chunk_size = int(self.config.evaluation_negative_batch_size)
        for start in range(0, flow.shape[0], chunk_size):
            end = min(start + chunk_size, flow.shape[0])
            exit_chunk = None if exit_flow is None else exit_flow[start:end]
            outputs.append(self.target_model.forward(flow[start:end], exit_flow=exit_chunk).to(self.trainable_device))
        return self.torch.cat(outputs, dim=0)

    @staticmethod
    def _classification_summary(totals: dict[str, float], prefix: str) -> dict[str, float | int | bool]:
        tp = totals[f"{prefix}_tp"]
        fp = totals[f"{prefix}_fp"]
        tn = totals[f"{prefix}_tn"]
        fn = totals[f"{prefix}_fn"]
        available = (tp + fp + tn + fn) > 0
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        return {
            f"{prefix}_classification_available": available,
            f"{prefix}_precision": float(precision),
            f"{prefix}_recall": float(recall),
            f"{prefix}_f1": float(2.0 * precision * recall / max(1e-12, precision + recall)),
            f"{prefix}_fpr": float(fp / max(1.0, fp + tn)),
            f"{prefix}_tp": int(tp),
            f"{prefix}_fp": int(fp),
            f"{prefix}_tn": int(tn),
            f"{prefix}_fn": int(fn),
        }

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
