from __future__ import annotations

import unittest

try:
    import torch

    from second_workpoint.training.deepcoffea_protocol import (
        aggregate_session_scores,
        partition_sessions_by_ipd,
    )
except ImportError:  # pragma: no cover - lightweight environments may omit torch
    torch = None
    aggregate_session_scores = None
    partition_sessions_by_ipd = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DeepCoffeaProtocolTests(unittest.TestCase):
    def test_partition_produces_work1_window_layout_and_preserves_gradient(self):
        session = torch.stack(
            [
                torch.full((12,), 1000.0),
                torch.arange(1, 13, dtype=torch.float32),
            ]
        ).requires_grad_()

        windows = partition_sessions_by_ipd(
            [session],
            delta_seconds=1.0,
            window_seconds=3.0,
            window_count=3,
            packet_limit=4,
        )

        self.assertEqual(tuple(windows.shape), (1, 3, 8))
        torch.testing.assert_close(windows[0, :, 0], torch.zeros(3))
        work1_mask = torch.sigmoid(torch.tensor([30.0, 20.0, 10.0, 0.0]))
        expected_sizes = torch.stack(
            [
                torch.tensor([1.0, 2.0, 3.0, 4.0]) * work1_mask,
                torch.tensor([3.0, 4.0, 5.0, 6.0]) * work1_mask,
                torch.tensor([5.0, 6.0, 7.0, 8.0]) * work1_mask,
            ]
        )
        torch.testing.assert_close(windows[0, :, 4:], expected_sizes)
        windows.sum().backward()
        self.assertGreater(float(session.grad.abs().sum()), 0.0)

    def test_vote_score_matches_nine_of_eleven_decision(self):
        similarities = torch.tensor(
            [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.0]]
        )
        score = aggregate_session_scores(similarities, vote_threshold=9)
        torch.testing.assert_close(score, torch.tensor([0.15]))
        self.assertFalse(bool((score >= 0.2).item()))


if __name__ == "__main__":
    unittest.main()
