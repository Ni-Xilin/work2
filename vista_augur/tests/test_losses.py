from __future__ import annotations

import unittest

import numpy as np

from second_workpoint.config import ExperimentConfig
from second_workpoint.training.losses import TargetedOverheadLoss


class LossTests(unittest.TestCase):
    def make_config(self, target_model: str):
        return ExperimentConfig(
            target_model=target_model,
            backbone_model_path="",
            time_channel_indices=[0],
            size_channel_indices=[1],
            enc_in=2,
        )

    def test_deepcorr_uses_targeted_bce(self):
        criterion = TargetedOverheadLoss(self.make_config("Deepcorr300"))
        flow = np.ones((2, 2, 4), dtype=np.float32)
        low = criterion.forward(np.asarray([[-2.0], [-2.0]]), np.zeros((2, 1)), flow, flow)
        high = criterion.forward(np.asarray([[2.0], [2.0]]), np.zeros((2, 1)), flow, flow)
        self.assertLess(low["label_loss"], high["label_loss"])

    def test_deepcoffea_uses_margin_hinge(self):
        config = self.make_config("Deepcoffea")
        config.deepcoffea_similarity_margin = -0.5
        criterion = TargetedOverheadLoss(config)
        flow = np.ones((2, 2, 4), dtype=np.float32)
        result = criterion.forward(np.asarray([[-0.7], [0.1]]), None, flow, flow)
        self.assertAlmostEqual(result["label_loss"], 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
