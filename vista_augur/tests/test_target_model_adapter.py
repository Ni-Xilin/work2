from __future__ import annotations

import unittest

try:
    import torch

    from second_workpoint.models.target_model_adapter import (
        _mdeepcorr_cascade_logits,
        _prepare_deepcoffea_flat_flow_torch,
    )
except ImportError:  # pragma: no cover - local lightweight environments may omit torch
    torch = None
    _mdeepcorr_cascade_logits = None
    _prepare_deepcoffea_flat_flow_torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MDeepCorrCascadeTests(unittest.TestCase):
    def test_stage1_rejects_low_scores_and_stage2_replaces_passed_scores(self):
        stage1_scores = torch.tensor([[0.005], [0.02]], requires_grad=True)
        stage2_scores = torch.tensor([[0.9], [0.3]], requires_grad=True)
        logits = _mdeepcorr_cascade_logits(
            torch.logit(stage1_scores),
            torch.logit(stage2_scores),
            threshold=0.01,
            torch_module=torch,
        )

        torch.testing.assert_close(torch.sigmoid(logits), torch.tensor([[0.005], [0.3]]))
        logits.sum().backward()
        self.assertNotEqual(float(stage1_scores.grad[0]), 0.0)
        self.assertEqual(float(stage1_scores.grad[1]), 0.0)
        self.assertEqual(float(stage2_scores.grad[0]), 0.0)
        self.assertNotEqual(float(stage2_scores.grad[1]), 0.0)

    def test_deepcoffea_accepts_prepartitioned_session_windows(self):
        windows = torch.arange(2 * 11 * 1000, dtype=torch.float32).reshape(2, 11, 1000)
        flattened = _prepare_deepcoffea_flat_flow_torch(
            windows,
            target_length=500,
            torch_module=torch,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(flattened.shape), (22, 1000))
        torch.testing.assert_close(flattened, windows.reshape(22, 1000))


if __name__ == "__main__":
    unittest.main()
