"""DeepCoFFEA-specific session partitioning and vote aggregation."""

from __future__ import annotations


def partition_sessions_by_ipd(
    sessions,
    delta_seconds: float,
    window_seconds: float,
    window_count: int,
    packet_limit: int,
    mask_steepness: float = 10.0,
):
    """Repartition perturbed Tor sessions with Work1's differentiable routine."""

    if not sessions:
        raise ValueError("DeepCoFFEA partitioning requires at least one session.")
    torch = _import_torch()
    window_ms = float(window_seconds) * 1000.0
    delta_ms = float(delta_seconds) * 1000.0
    offset_ms = window_ms - delta_ms
    partitioned = []
    differentiable_search = _work1_differentiable_searchsorted(torch)
    base_grid = torch.arange(packet_limit, dtype=torch.float32, device=sessions[0].device)

    for session in sessions:
        if session.ndim != 2 or session.shape[0] < 2:
            raise ValueError(f"DeepCoFFEA session must have shape (2, length), got {tuple(session.shape)}")
        if session.shape[1] == 0:
            raise ValueError("DeepCoFFEA session cannot be empty.")

        ipd = session[0]
        size = session[1]
        cumulative_time = ipd.abs().cumsum(dim=0)
        session_windows = []

        for window_index in range(int(window_count)):
            start_time = torch.tensor(float(window_index * offset_ms), device=session.device)
            end_time = torch.tensor(float(start_time + window_ms), device=session.device)
            start_index = differentiable_search.apply(cumulative_time, start_time)
            end_index = differentiable_search.apply(cumulative_time, end_time)
            sample_indices = start_index + base_grid

            window_ipd = _work1_differentiable_sample_1d(ipd, sample_indices, torch)
            window_size = _work1_differentiable_sample_1d(size, sample_indices, torch)
            valid_mask = torch.sigmoid(
                (end_index - sample_indices) * float(mask_steepness)
            )
            window_ipd = window_ipd * valid_mask
            window_size = window_size * valid_mask

            zero_mask = torch.ones_like(window_ipd)
            zero_mask[0] = 0.0
            window_ipd = window_ipd * zero_mask
            window = torch.cat([window_ipd, window_size])
            if window.shape[0] < packet_limit * 2:
                padding = torch.zeros(packet_limit * 2 - window.shape[0], device=session.device)
                window = torch.cat([window, padding])
            else:
                window = window[: packet_limit * 2]
            session_windows.append(window)

        partitioned.append(torch.stack(session_windows, dim=0))

    return torch.stack(partitioned, dim=0)


def _work1_differentiable_searchsorted(torch):
    """Return the custom autograd search used verbatim by Work1's train path."""

    class DifferentiableSearchsorted(torch.autograd.Function):
        @staticmethod
        def forward(ctx, sorted_sequence, values):
            indices = torch.searchsorted(sorted_sequence, values, right=True)
            indices.clamp_(0, sorted_sequence.size(-1) - 1)
            ctx.save_for_backward(sorted_sequence, values, indices)
            return indices

        @staticmethod
        def backward(ctx, grad_output):
            sorted_sequence, values, indices = ctx.saved_tensors
            grad_sorted_sequence = torch.zeros_like(sorted_sequence)
            idx_left = (indices - 1).clamp(0, sorted_sequence.size(-1) - 1)
            idx_right = indices.clamp(0, sorted_sequence.size(-1) - 1)
            val_left = sorted_sequence.gather(-1, idx_left)
            val_right = sorted_sequence.gather(-1, idx_right)
            weight_right = (values - val_left) / (val_right - val_left + 1e-9)
            weight_left = 1 - weight_right
            grad_sorted_sequence.scatter_add_(-1, idx_left, grad_output * weight_left)
            grad_sorted_sequence.scatter_add_(-1, idx_right, grad_output * weight_right)
            return grad_sorted_sequence, None

    return DifferentiableSearchsorted


def _work1_differentiable_sample_1d(data, query_indices, torch):
    """Linearly sample one Work1 session channel at the requested indices."""

    length = data.size(0)
    idx_floor = query_indices.floor().long()
    idx_ceil = idx_floor + 1
    idx_floor_clamped = idx_floor.clamp(0, length - 1)
    idx_ceil_clamped = idx_ceil.clamp(0, length - 1)
    val_floor = data[idx_floor_clamped]
    val_ceil = data[idx_ceil_clamped]
    alpha = query_indices - idx_floor.float()
    sampled_data = val_floor * (1 - alpha) + val_ceil * alpha
    mask = ((query_indices >= 0) & (query_indices < length - 1)).float().detach()
    return sampled_data * mask


def aggregate_session_scores(window_similarities, vote_threshold: int):
    """Return the similarity threshold equivalent to a k-of-n window vote."""

    if window_similarities.ndim != 2:
        raise ValueError(
            "DeepCoFFEA window similarities must have shape (sessions, windows), "
            f"got {tuple(window_similarities.shape)}"
        )
    window_count = int(window_similarities.shape[1])
    if vote_threshold <= 0 or vote_threshold > window_count:
        raise ValueError("DeepCoFFEA vote_threshold must be in [1, window_count].")
    return window_similarities.topk(int(vote_threshold), dim=1).values[:, -1]


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("DeepCoFFEA protocol requires PyTorch.") from exc
    return torch
