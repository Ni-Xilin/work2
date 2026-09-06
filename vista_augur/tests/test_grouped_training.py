from __future__ import annotations

import unittest

from second_workpoint.config import ExperimentConfig

try:
    import torch

    from second_workpoint.training.losses import TargetedOverheadLoss
    from second_workpoint.training.trainer import TorchTrainer
except ImportError:  # pragma: no cover - local lightweight environments may omit torch
    torch = None
    TargetedOverheadLoss = None
    TorchTrainer = None


class _FakeGenerator:
    def __init__(self):
        self.last_perturbation = None

    def forward(self, history_seq, prompt_text):
        self.last_perturbation = torch.full(
            (history_seq.shape[0], 2, 4),
            0.1,
            dtype=history_seq.dtype,
            device=history_seq.device,
            requires_grad=True,
        )
        return {"perturbation": self.last_perturbation}


class _FakeTarget:
    def __init__(self):
        self.calls = []

    def forward(self, flow, exit_flow=None):
        self.calls.append(flow.detach().clone())
        return flow.mean(dim=tuple(range(1, flow.ndim))).reshape(-1, 1)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GroupedTrainingTests(unittest.TestCase):
    def test_all_windows_are_written_before_target_feedback(self):
        config = ExperimentConfig(
            backbone_model_path="",
            target_model="Deepcorr300",
            flow_size=8,
            seq_len=4,
            pred_len=2,
            patch_len=2,
            stride=2,
            enc_in=4,
            trainable_width=16,
            d_model=16,
            n_heads=4,
            time_channel_indices=[0, 1],
            size_channel_indices=[2, 3],
            tor_row_indices=[0, 3, 4, 7],
        )
        trainer = TorchTrainer.__new__(TorchTrainer)
        trainer.config = config
        trainer.torch = torch
        trainer.trainable_device = torch.device("cpu")
        trainer.model = _FakeGenerator()
        trainer.target_model = _FakeTarget()
        trainer.criterion = TargetedOverheadLoss(config)

        signed_future = torch.tensor(
            [[[[1.0, -2.0, 0.0, 3.0], [1.0, -2.0, 0.0, 3.0]],
              [[1.0, -2.0, 0.0, 3.0], [1.0, -2.0, 0.0, 3.0]]]]
        )
        batch = {
            "full_flow": torch.ones(1, 4, 8),
            "history_seq": torch.ones(1, 2, 4, 4),
            "clean_future": signed_future,
            "prompt_text": [["first", "second"]],
            "future_mask": torch.ones(1, 2, 2),
            "writeback_meta": torch.tensor([[[4, 2], [6, 2]]]),
            "target_full_flow": torch.ones(1, 8, 8),
            "target_negative_flow": torch.ones(1, 1, 8, 8),
        }

        outputs = trainer._forward_batch(batch, log_shapes=False)
        outputs["loss"].backward()

        adversarial_target_flow = trainer.target_model.calls[1]
        expected_by_row = {
            0: 1.1,
            3: -2.1,
            4: 0.0,
            7: 3.1,
        }
        for row, expected in expected_by_row.items():
            torch.testing.assert_close(
                adversarial_target_flow[0, row, 4:8],
                torch.full((4,), expected),
            )
        self.assertEqual(outputs["batch_weight"], 1)
        self.assertGreater(float(trainer.model.last_perturbation.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
