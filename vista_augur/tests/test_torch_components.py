from __future__ import annotations

import unittest

from second_workpoint.config import ExperimentConfig

try:
    import torch

    from second_workpoint.models.torch_real_model import VisualPatchAdapter
except ImportError:  # pragma: no cover - local lightweight environments may omit torch
    torch = None
    VisualPatchAdapter = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchComponentTests(unittest.TestCase):
    def test_visual_adapter_is_differentiable_and_preserves_patch_count(self):
        config = ExperimentConfig(
            backbone_model_path="",
            seq_len=8,
            pred_len=4,
            patch_len=2,
            flow_size=12,
            enc_in=4,
            trainable_width=16,
            d_model=16,
            n_heads=4,
            visual_conv_channels=4,
        )
        adapter = VisualPatchAdapter(config)
        flow = torch.randn(3, 4, 8, requires_grad=True)

        output = adapter(flow)

        self.assertEqual(tuple(output.shape), (3, 4, 16))
        output[..., 0].sum().backward()
        self.assertIsNotNone(flow.grad)
        self.assertGreater(float(flow.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
