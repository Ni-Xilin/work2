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
    """Repartition perturbed Tor sessions using Work1's IPD-time protocol.

    The selected packet values remain differentiable.  As in Work1, discrete
    ``searchsorted`` boundaries are treated as fixed during backpropagation.
    """

    if not sessions:
        raise ValueError("DeepCoFFEA partitioning requires at least one session.")
    torch = _import_torch()
    offset_ms = (float(window_seconds) - float(delta_seconds)) * 1000.0
    window_ms = float(window_seconds) * 1000.0
    partitioned = []

    for session in sessions:
        if session.ndim != 2 or session.shape[0] < 2:
            raise ValueError(f"DeepCoFFEA session must have shape (2, length), got {tuple(session.shape)}")
        if session.shape[1] == 0:
            raise ValueError("DeepCoFFEA session cannot be empty.")

        ipd = session[0]
        size = session[1]
        cumulative_time = ipd.abs().cumsum(dim=0)
        base_indices = torch.arange(packet_limit, device=session.device, dtype=torch.long)
        session_windows = []

        for window_index in range(int(window_count)):
            start_time = torch.tensor(window_index * offset_ms, device=session.device, dtype=session.dtype)
            end_time = torch.tensor(window_index * offset_ms + window_ms, device=session.device, dtype=session.dtype)
            start_index = torch.searchsorted(cumulative_time.detach(), start_time, right=True)
            end_index = torch.searchsorted(cumulative_time.detach(), end_time, right=True)
            sample_indices = start_index + base_indices
            clamped_indices = sample_indices.clamp(max=session.shape[1] - 1)
            # Work1's linear sampler requires a right neighbour, so its mask
            # intentionally excludes the final packet index.
            in_session = sample_indices < max(0, session.shape[1] - 1)
            within_time_window = torch.sigmoid(
                (end_index.to(session.dtype) - sample_indices.to(session.dtype)) * float(mask_steepness)
            )
            valid_mask = in_session.to(session.dtype) * within_time_window

            window_ipd = ipd[clamped_indices] * valid_mask
            window_size = size[clamped_indices] * valid_mask
            first_ipd_mask = torch.ones_like(window_ipd)
            first_ipd_mask[0] = 0.0
            session_windows.append(torch.cat([window_ipd * first_ipd_mask, window_size], dim=0))

        partitioned.append(torch.stack(session_windows, dim=0))

    return torch.stack(partitioned, dim=0)


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
