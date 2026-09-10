"""Shared implementation for the three Work2 checkpoint evaluators."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VISTA_ROOT = REPOSITORY_ROOT / "vista_augur"
for import_root in (REPOSITORY_ROOT, VISTA_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from second_workpoint.config import load_config  # noqa: E402
from second_workpoint.training.trainer import build_trainer  # noqa: E402
from second_workpoint.utils.runtime import seed_everything  # noqa: E402


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    config: str
    checkpoint: str
    model_dir: str
    target_names: tuple[str, ...]
    protocol: str
    batch_size: int
    num_workers: int
    shuffle: bool
    drop_last: bool
    max_samples: int = 0
    max_steps: int = 0
    scoring_batch_size: int = 256
    negative_pair_seed: int | None = None


def run(profile: EvaluationProfile) -> None:
    args = _arguments(profile)
    config_path = args.config.resolve()
    config = load_config(config_path)
    if config.target_model.lower() not in profile.target_names:
        raise ValueError(
            f"{profile.name} evaluator cannot evaluate target_model={config.target_model!r}; "
            f"expected one of {profile.target_names}."
        )

    config.run_mode = "evaluate"
    config.is_training = 0
    config.eval_split = args.eval_split
    config.final_eval_split = args.eval_split
    config.resume_from_checkpoint = _resolve_checkpoint_argument(args.checkpoint, profile)
    config.batch_size = args.batch_size
    config.max_eval_samples = args.max_samples
    config.max_eval_steps = args.max_steps
    config.num_workers = args.num_workers
    config.evaluation_negative_batch_size = profile.scoring_batch_size
    if args.negative_pairs is not None:
        config.negative_pairs_per_sample = args.negative_pairs
    physical_gpus = _parse_gpu_pair(args.gpus)
    _assert_gpus_idle(physical_gpus, allow_busy=not args.require_idle_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in physical_gpus)
    config.visible_gpu_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    config.backbone_device = "cuda:0"
    config.backbone_secondary_device = ""
    config.backbone_split_layer_index = 0
    config.trainable_device = "cuda:1"
    config.target_device = "cuda:1"
    config.validate()
    seed_everything(config.random_seed)

    output_dir = (args.output_dir if args.output_dir is not None else Path(profile.model_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        trainer = build_trainer(config)
    _reset_eval_loader(trainer, profile, args)
    if profile.protocol == "deepcoffea_matrix":
        print(
            f"Number of (session)examples to test: {len(trainer.eval_dataset)}, "
            f"number of windows: {config.deepcoffea_n_windows}"
        )
        sequence_arrays = _collect_deepcoffea_sequences(trainer)
        score_arrays = _evaluate_deepcoffea_sequences(trainer, sequence_arrays)
        _save_deepcoffea_work1_outputs(output_dir, sequence_arrays, score_arrays)
    else:
        sequence_arrays = _collect_deepcorr_sequences(trainer)
        _print_work1_test_results(sequence_arrays)
        _save_deepcorr_work1_samples(output_dir, sequence_arrays, profile, torch_module=trainer.torch)
        _print_work1_scoring_preamble(trainer, profile)
        score_arrays = _evaluate_deepcorr_sequences(trainer, sequence_arrays, profile)
        _save_deepcorr_work1_scores(output_dir, score_arrays, profile)


def _arguments(profile: EvaluationProfile) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Evaluate a trained Work2 {profile.name} generator and export plotting-ready data."
    )
    parser.add_argument("--config", type=Path, default=Path(profile.config))
    parser.add_argument(
        "--checkpoint",
        default=profile.checkpoint,
        help=(
            "Generator checkpoint path, best/latest relative to the configured checkpoint directory, "
            "or auto to use the only .pt/.pth file beside this evaluator."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory. Default: {profile.model_dir}/.",
    )
    parser.add_argument(
        "--gpus",
        default="2,3",
        help="Two physical GPUs as BACKBONE,GENERATOR_TARGET. Default: 2,3.",
    )
    parser.add_argument(
        "--require-idle-gpus",
        action="store_true",
        help="Refuse to start unless both selected GPUs have no compute processes.",
    )
    parser.add_argument("--eval-split", choices=("test", "val"), default="test")
    parser.add_argument("--batch-size", type=int, default=profile.batch_size)
    parser.add_argument("--num-workers", type=int, default=profile.num_workers)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=profile.max_samples,
        help="Work1 default for this evaluator; positive overrides are for smoke tests.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=profile.max_steps,
        help="Work1 default for this evaluator; positive overrides are for smoke tests.",
    )
    parser.add_argument(
        "--negative-pairs",
        type=int,
        default=None,
        help="Override mismatches per session for DeepCorr smoke tests.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_samples < 0 or args.max_steps < 0:
        parser.error("batch-size must be positive; num-workers, max-samples and max-steps must be non-negative")
    if args.negative_pairs is not None and args.negative_pairs <= 0:
        parser.error("negative-pairs must be positive")
    return args


def _resolve_checkpoint_argument(value: str, profile: EvaluationProfile) -> str:
    if str(value).strip().lower() != "auto":
        return str(value)
    model_dir = (REPOSITORY_ROOT / profile.model_dir).resolve()
    candidates = sorted(
        path
        for pattern in ("*.pt", "*.pth")
        for path in model_dir.glob(pattern)
        if path.is_file()
    )
    if len(candidates) == 1:
        return str(candidates[0])
    if not candidates:
        raise FileNotFoundError(
            f"No .pt/.pth generator checkpoint found in {model_dir}. "
            "Place one checkpoint there or pass --checkpoint explicitly."
        )
    names = ", ".join(path.name for path in candidates)
    raise RuntimeError(
        f"Multiple checkpoints found in {model_dir}: {names}. "
        "Pass --checkpoint with the intended file so evaluation cannot select the wrong model."
    )


def _reset_eval_loader(trainer, profile: EvaluationProfile, args: argparse.Namespace) -> None:
    """Rebuild the test loader with the per-target Work1 evaluation settings."""

    from torch.utils.data import DataLoader

    trainer.eval_loader = DataLoader(
        trainer.eval_dataset,
        batch_size=args.batch_size,
        shuffle=profile.shuffle,
        num_workers=args.num_workers,
        collate_fn=trainer._torch_collate_batch,
        drop_last=profile.drop_last,
    )


def _collect_deepcorr_sequences(trainer) -> dict[str, np.ndarray]:
    original, adversarial, masks, sample_keys = [], [], [], []
    trainer.model.eval()
    with trainer.torch.no_grad():
        for step, batch in enumerate(trainer.eval_loader, start=1):
            outputs = trainer._forward_batch(batch, log_shapes=False, include_negatives=False, collect_artifacts=True)
            original.append(outputs["artifact_target_original_flow"].numpy())
            adversarial.append(outputs["artifact_target_adv_flow"].numpy())
            masks.append(outputs["artifact_flow_mask"].numpy())
            sample_keys.extend(str(value) for value in batch["sample_key"])
            if trainer.config.max_eval_steps > 0 and step >= trainer.config.max_eval_steps:
                break
    if not sample_keys:
        raise RuntimeError("Evaluation split produced no samples.")
    original_samples = np.concatenate(original).astype(np.float32)[:, None]
    adv_samples = np.concatenate(adversarial).astype(np.float32)[:, None]
    valid_mask = np.concatenate(masks).astype(np.uint8)[:, None]
    tor_rows = list(trainer.config.tor_row_indices)
    time_channels = [tor_rows[index] for index in trainer.config.time_channel_indices]
    size_channels = [tor_rows[index] for index in trainer.config.size_channel_indices]
    return {
        # Work1-compatible names and full [N,1,8,L] target-model representation.
        "original_samples": original_samples,
        "adv_samples": adv_samples,
        "valid_mask": valid_mask,
        "time_ratios": np.asarray(
            _mean_relative_l2(original_samples, adv_samples, time_channels, valid_mask), dtype=np.float32
        ),
        "size_ratios": np.asarray(
            _mean_relative_l2(original_samples, adv_samples, size_channels, valid_mask), dtype=np.float32
        ),
        "sample_keys": np.asarray(sample_keys, dtype=np.str_),
        "tor_row_indices": np.asarray(trainer.config.tor_row_indices, dtype=np.int64),
    }


def _evaluate_deepcorr_sequences(trainer, sequence_arrays, profile: EvaluationProfile) -> dict[str, np.ndarray]:
    adversarial = np.asarray(sequence_arrays["adv_samples"][:, 0], dtype=np.float32)
    negative_count = int(trainer.config.negative_pairs_per_sample)
    if profile.name == "mdeepcorr":
        tor_indices, exit_indices = _work1_mdeepcorr_negative_pair_indices(
            adversarial.shape[0], negative_count
        )
    else:
        tor_indices, exit_indices = _work1_negative_pair_indices(
            adversarial.shape[0], negative_count, profile.negative_pair_seed
        )
    adv_positive = _score_deepcorr_flows(trainer, adversarial)
    adv_negative = _score_deepcorr_pairs(trainer, adversarial, tor_indices, exit_indices)
    interleave = _work1_positive_first_scores if profile.name == "mdeepcorr" else _work1_interleaved_scores
    adv_all_outputs, all_labels = interleave(adv_positive, adv_negative, negative_count)
    if profile.name == "deepcorr300":
        retained = (adv_all_outputs.size // profile.scoring_batch_size) * profile.scoring_batch_size
        adv_all_outputs = adv_all_outputs[:retained]
        all_labels = all_labels[:retained]
    return {
        "adv_all_outputs": adv_all_outputs,
        "all_labels": all_labels,
    }


def _work1_mdeepcorr_negative_pair_indices(session_count: int, negative_count: int):
    if session_count < 2 or negative_count > session_count - 1:
        raise ValueError("negative_pairs_per_sample must be smaller than the evaluated session count")
    pool = np.arange(session_count, dtype=np.int64)
    tor_indices, exit_indices = [], []
    for tor_index in range(session_count):
        rng = np.random.RandomState(tor_index)
        rng.shuffle(pool)
        selected = pool[pool != tor_index][:negative_count]
        tor_indices.extend([tor_index] * negative_count)
        exit_indices.extend(selected.tolist())
    return np.asarray(tor_indices, dtype=np.int64), np.asarray(exit_indices, dtype=np.int64)


def _work1_negative_pair_indices(session_count: int, negative_count: int, seed: int | None):
    if session_count < 2 or negative_count > session_count - 1:
        raise ValueError("negative_pairs_per_sample must be smaller than the evaluated session count")
    # The Work1 DeepCorr scorer did not set a seed; RandomState(None) retains
    # that behavior. Tests may pass an explicit seed to lock structural checks.
    rng = np.random.RandomState(seed)
    pool = np.arange(session_count, dtype=np.int64)
    tor_indices = []
    exit_indices = []
    for exit_index in range(session_count):
        shuffled = pool.copy()
        rng.shuffle(shuffled)
        selected = shuffled[shuffled != exit_index][:negative_count]
        tor_indices.extend(selected.tolist())
        exit_indices.extend([exit_index] * negative_count)
    return np.asarray(tor_indices, dtype=np.int64), np.asarray(exit_indices, dtype=np.int64)


def _work1_interleaved_scores(positive_scores, negative_scores, negative_count: int):
    positives = np.asarray(positive_scores, dtype=np.float32).reshape(-1)
    negatives = np.asarray(negative_scores, dtype=np.float32).reshape(positives.size, negative_count)
    outputs = np.concatenate((negatives, positives[:, None]), axis=1).reshape(-1)
    labels = np.zeros((positives.size, negative_count + 1), dtype=np.float32)
    labels[:, -1] = 1
    return outputs.astype(np.float32), labels.reshape(-1)


def _work1_positive_first_scores(positive_scores, negative_scores, negative_count: int):
    positives = np.asarray(positive_scores, dtype=np.float32).reshape(-1)
    negatives = np.asarray(negative_scores, dtype=np.float32).reshape(positives.size, negative_count)
    outputs = np.concatenate((positives[:, None], negatives), axis=1).reshape(-1)
    labels = np.zeros((positives.size, negative_count + 1), dtype=np.float32)
    labels[:, 0] = 1
    return outputs.astype(np.float32), labels.reshape(-1)


def _score_deepcorr_flows(trainer, flows: np.ndarray) -> np.ndarray:
    torch = trainer.torch
    scores = []
    chunk_size = int(trainer.config.evaluation_negative_batch_size)
    with torch.no_grad():
        for start in range(0, flows.shape[0], chunk_size):
            tensor = torch.from_numpy(flows[start : start + chunk_size]).to(
                trainer.trainable_device, dtype=torch.float32
            )
            logits = trainer.target_model.forward(tensor).reshape(-1)
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def _score_deepcorr_pairs(trainer, flows: np.ndarray, tor_indices, exit_indices) -> np.ndarray:
    scores = []
    chunk_size = int(trainer.config.evaluation_negative_batch_size)
    tor_rows = np.asarray(trainer.config.tor_row_indices, dtype=np.int64)
    for start in range(0, tor_indices.size, chunk_size):
        end = min(start + chunk_size, tor_indices.size)
        pairs = flows[exit_indices[start:end]].copy()
        pairs[:, tor_rows, :] = flows[tor_indices[start:end]][:, tor_rows, :]
        scores.append(_score_deepcorr_flows(trainer, pairs))
    return np.concatenate(scores).astype(np.float32)


def _collect_deepcoffea_sequences(trainer) -> dict[str, np.ndarray]:
    clean_windows, adv_windows, exit_windows = [], [], []
    original_batches, adv_batches, mask_batches, sample_keys = [], [], [], []
    trainer.model.eval()
    with trainer.torch.no_grad():
        for step, batch in enumerate(trainer.eval_loader, start=1):
            outputs = trainer._forward_batch(batch, log_shapes=False, include_negatives=False, collect_artifacts=True)
            clean_windows.append(outputs["artifact_clean_tor_windows"].numpy())
            adv_windows.append(outputs["artifact_adv_tor_windows"].numpy())
            exit_windows.append(outputs["artifact_exit_windows"].numpy())
            original_batches.append(outputs["artifact_original_flow"].numpy())
            adv_batches.append(outputs["artifact_adv_flow"].numpy())
            mask_batches.append(outputs["artifact_flow_mask"].numpy())
            sample_keys.extend(str(value) for value in batch["sample_key"])
            indices = [int(value.rsplit(":", 1)[-1]) for value in batch["sample_key"]]
            print(
                "Batch: {0},session_idx:{1} \nTime L2 distance rate: {2}, Size L2 Distance rate: {3}".format(
                    step - 1, indices, outputs["time_ratio"], outputs["size_ratio"]
                )
            )
            print(f"{len(sample_keys)}session's pairs done.")
            if trainer.config.max_eval_steps > 0 and step >= trainer.config.max_eval_steps:
                break
    if len(sample_keys) < 2:
        raise RuntimeError("DeepCoFFEA matrix evaluation requires at least two sessions.")

    clean = np.concatenate(clean_windows).astype(np.float32)
    adversarial = np.concatenate(adv_windows).astype(np.float32)
    exits = np.concatenate(exit_windows).astype(np.float32)
    original, defended, valid_mask = _pad_variable_flows(original_batches, adv_batches, mask_batches)
    time_channels = list(trainer.config.time_channel_indices)
    size_channels = list(trainer.config.size_channel_indices)
    return {
        "original_sessions": original,
        "adv_sessions": defended,
        "valid_mask": valid_mask,
        "time_ratios": np.asarray(
            _mean_relative_l2(original, defended, time_channels, valid_mask), dtype=np.float32
        ),
        "size_ratios": np.asarray(
            _mean_relative_l2(original, defended, size_channels, valid_mask), dtype=np.float32
        ),
        "clean_tor_windows": clean,
        "adv_tor_windows": adversarial,
        "exit_windows": exits,
        "sample_keys": np.asarray(sample_keys, dtype=np.str_),
        "n_windows": np.asarray(trainer.config.deepcoffea_n_windows, dtype=np.int32),
        "vote_threshold": np.asarray(trainer.config.deepcoffea_vote_threshold, dtype=np.int32),
        "partition_protocol": np.asarray("work1_hard_left_compact", dtype=np.str_),
    }


def _evaluate_deepcoffea_sequences(trainer, sequence_arrays) -> dict[str, np.ndarray]:
    clean = np.asarray(sequence_arrays["clean_tor_windows"], dtype=np.float32)
    adversarial = np.asarray(sequence_arrays["adv_tor_windows"], dtype=np.float32)
    exits = np.asarray(sequence_arrays["exit_windows"], dtype=np.float32)
    clean_embedding = _embed_windows(trainer, clean, branch="anchor")
    adv_embedding = _embed_windows(trainer, adversarial, branch="anchor")
    exit_embedding = _embed_windows(trainer, exits, branch="pandn")
    clean_work1_matrix = _work1_full_cosine_matrix(clean_embedding, exit_embedding)
    adv_work1_matrix = _work1_full_cosine_matrix(adv_embedding, exit_embedding)
    return {
        "clean_work1_matrix": clean_work1_matrix,
        "adv_work1_matrix": adv_work1_matrix,
    }


def _embed_windows(trainer, windows: np.ndarray, branch: str) -> np.ndarray:
    torch = trainer.torch
    model = getattr(trainer.target_model, branch)
    flat = windows.reshape(-1, windows.shape[-1])
    chunks = []
    chunk_size = int(trainer.config.evaluation_negative_batch_size)
    device = trainer.target_model.device
    with torch.no_grad():
        for start in range(0, flat.shape[0], chunk_size):
            tensor = torch.from_numpy(flat[start : start + chunk_size]).to(device=device, dtype=torch.float32)
            embedding = model(tensor)
            if embedding.ndim == 1:
                embedding = embedding.unsqueeze(0)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
            chunks.append(embedding.float().cpu().numpy())
    return np.concatenate(chunks).reshape(windows.shape[0], windows.shape[1], -1)


def _window_matrices(tor_embedding: np.ndarray, exit_embedding: np.ndarray) -> np.ndarray:
    # Inputs are session-major [N,W,D]; output is the plotting protocol [W,N,N].
    return np.einsum("nwd,mwd->wnm", tor_embedding, exit_embedding, optimize=True).astype(np.float32)


def _work1_full_cosine_matrix(tor_embedding: np.ndarray, exit_embedding: np.ndarray) -> np.ndarray:
    """Return Work1's session-major, all-window cosine matrix."""

    tor_flat = tor_embedding.reshape(-1, tor_embedding.shape[-1])
    exit_flat = exit_embedding.reshape(-1, exit_embedding.shape[-1])
    return np.matmul(tor_flat, exit_flat.T).astype(np.float32)


def _pad_variable_flows(original_batches, adv_batches, mask_batches):
    originals = [flow for batch in original_batches for flow in batch]
    adversarials = [flow for batch in adv_batches for flow in batch]
    masks = [mask for batch in mask_batches for mask in batch]
    max_length = max(flow.shape[-1] for flow in originals)
    clean = np.zeros((len(originals), originals[0].shape[-2], max_length), dtype=np.float32)
    defended = np.zeros_like(clean)
    valid = np.zeros((len(originals), max_length), dtype=np.uint8)
    for index, (source, adv, mask) in enumerate(zip(originals, adversarials, masks)):
        length = source.shape[-1]
        clean[index, :, :length] = source
        defended[index, :, :length] = adv
        valid[index, :length] = mask[:length] > 0
    return clean, defended, valid


def _mean_relative_l2(original, adversarial, channels, valid_mask) -> float:
    clean = np.asarray(original, dtype=np.float64)
    defended = np.asarray(adversarial, dtype=np.float64)
    selected_clean = np.take(np.abs(clean), channels, axis=-2)
    selected_delta = np.take(np.abs(defended) - np.abs(clean), channels, axis=-2)
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim == clean.ndim - 1:
        mask = np.expand_dims(mask, axis=-2)
    selected_mask = np.broadcast_to(mask, clean.shape)
    selected_mask = np.take(selected_mask, channels, axis=-2)
    selected_clean = np.where(selected_mask, selected_clean, 0.0)
    selected_delta = np.where(selected_mask, selected_delta, 0.0)
    # Work1 computes one ratio per selected channel, then averages over channels,
    # sessions, and batches. Equal-size drop-last batches make this global mean
    # exactly equivalent to its mean-of-batch-means implementation.
    numerators = np.linalg.norm(selected_delta, axis=-1)
    denominators = np.linalg.norm(selected_clean, axis=-1)
    return float(np.mean(numerators / np.maximum(denominators, 1e-15)))


def _print_work1_test_results(sequence_arrays) -> None:
    print("\n--- Test Results ---")
    print(f"Average Time L2 Ratio: {float(sequence_arrays['time_ratios']):.4f}")
    print(f"Average Size L2 Ratio: {float(sequence_arrays['size_ratios']):.4f}")


def _save_deepcorr_work1_samples(
    output_dir, sequence_arrays, profile: EvaluationProfile, *, torch_module=None
) -> Path:
    if torch_module is None:
        import torch as torch_module
    prefix = "Gmdeepcorr" if profile.name == "mdeepcorr" else "Gdeepcorr"
    time_ratio = float(sequence_arrays["time_ratios"])
    size_ratio = float(sequence_arrays["size_ratios"])
    path = output_dir / f"{prefix}_advsamples_time{time_ratio:.4f}_size{size_ratio:.4f}.p"
    payload = {
        "adv_samples": torch_module.from_numpy(np.asarray(sequence_arrays["adv_samples"])),
        "original_samples": torch_module.from_numpy(np.asarray(sequence_arrays["original_samples"])),
        "time_ratios": time_ratio,
        "size_ratios": size_ratio,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def _print_work1_scoring_preamble(trainer, profile: EvaluationProfile) -> None:
    if profile.name == "deepcorr300":
        print(trainer.target_model.device)
        return
    print(f"Using device: {trainer.target_model.device}")
    print("Loading DC100 model...")
    print("DC100 model loaded.")
    print("Loading DC700 model...")
    print("DC700 model loaded.")
    print("\nPhase 1: Evaluating with DC100 (flow_size=100)...")


def _save_deepcorr_work1_scores(output_dir, score_arrays, profile: EvaluationProfile) -> Path:
    if profile.name == "mdeepcorr":
        threshold = 0.01
        path = output_dir / f"Gmdeepcorr_threshDC100_time0_{threshold:.4f}.p"
        payload = {"all_outputs": score_arrays["adv_all_outputs"], "all_labels": score_arrays["all_labels"]}
    else:
        path = output_dir / "Gtest_index300_time0_result.p"
        payload = (score_arrays["adv_all_outputs"], score_arrays["all_labels"])
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    if profile.name == "mdeepcorr":
        print(f"Evaluation results saved to {path}")
    return path


def _save_deepcoffea_work1_outputs(output_dir, sequence_arrays, score_arrays) -> tuple[Path, Path]:
    time_ratio = float(sequence_arrays["time_ratios"])
    size_ratio = float(sequence_arrays["size_ratios"])
    clean_matrix = score_arrays["clean_work1_matrix"]
    adv_matrix = score_arrays["adv_work1_matrix"]
    similarity = float(np.mean(np.diag(clean_matrix)))
    adv_similarity = float(np.mean(np.diag(adv_matrix)))
    clean_path = output_dir / f"corrmatrix_sim{similarity:.4f}.npz"
    adv_path = output_dir / (
        f"Gcorrmatrix_time{time_ratio:.4f}_size{size_ratio:.4f}_sim{adv_similarity:.4f}.npz"
    )
    np.savez_compressed(clean_path, corr_matrix=clean_matrix, sim=similarity)
    np.savez_compressed(
        adv_path,
        corr_matrix=adv_matrix,
        meantimeL2_rate=time_ratio,
        meansizeL2_rate=size_ratio,
        sim=adv_similarity,
    )
    print(f"Time L2 distance rate: {time_ratio}, Size L2 Distance rate: {size_ratio}")
    print(f"Cosine Similarity: {similarity}, G Cosine Similarity: {adv_similarity}")
    return clean_path, adv_path


def _parse_gpu_pair(value: str) -> tuple[int, int]:
    fields = [field.strip() for field in str(value).split(",") if field.strip()]
    if len(fields) != 2:
        raise ValueError("--gpus must contain exactly two physical indices, for example 2,3")
    try:
        pair = tuple(int(field) for field in fields)
    except ValueError as exc:
        raise ValueError("--gpus values must be non-negative integers") from exc
    if any(index < 0 for index in pair) or pair[0] == pair[1]:
        raise ValueError("--gpus must contain two distinct non-negative physical indices")
    return pair


def _assert_gpus_idle(gpu_indices, *, allow_busy: bool) -> None:
    """Refuse accidental evaluation/training colocation on either selected GPU."""

    busy = []
    for gpu_index in gpu_indices:
        command = [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("--gpus safety check requires nvidia-smi, but it was not found.") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout).strip()
            raise RuntimeError(f"Could not inspect physical GPU {gpu_index}: {detail}") from exc
        processes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if processes:
            busy.append(f"GPU {gpu_index}: " + "; ".join(processes))
    if busy and not allow_busy:
        raise RuntimeError(
            "Selected evaluation GPUs already have compute processes (" + " | ".join(busy) + "). "
            "Evaluation was not started, so it cannot contend with training. Wait for both GPUs to become idle; "
            "omit --require-idle-gpus only when resource contention is acceptable."
        )
    if busy:
        print(
            "[gpu warning] selected GPUs already have compute processes: "
            + " | ".join(busy)
            + ". Continuing because --require-idle-gpus was not specified."
        )
