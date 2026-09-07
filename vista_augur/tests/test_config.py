from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from second_workpoint.config import ExperimentConfig, load_config


class ExperimentConfigTests(unittest.TestCase):
    def make_config(self, **overrides):
        values = {
            "backbone_model_path": "",
            "backbone_model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "target_model_path": "",
        }
        values.update(overrides)
        return ExperimentConfig(**values)

    def test_portable_model_name_is_valid_without_local_model(self):
        self.make_config().validate()

    def test_rejects_disabled_numeric_branches(self):
        config = self.make_config(use_temporal_branch=False, use_visual_branch=False)
        with self.assertRaisesRegex(ValueError, "At least one"):
            config.validate()

    def test_rejects_invalid_deepcoffea_margin(self):
        config = self.make_config(
            deepcoffea_similarity_margin=0.3,
            deepcoffea_similarity_threshold=0.2,
        )
        with self.assertRaisesRegex(ValueError, "lower"):
            config.validate()

    def test_rejects_nonpositive_negative_pair_count(self):
        config = self.make_config(negative_pairs_per_sample=0)
        with self.assertRaisesRegex(ValueError, "negative_pairs_per_sample"):
            config.validate()

    def test_rejects_nonpositive_negative_evaluation_batch(self):
        config = self.make_config(evaluation_negative_batch_size=0)
        with self.assertRaisesRegex(ValueError, "evaluation_negative_batch_size"):
            config.validate()

    def test_rejects_invalid_mdeepcorr_stage1_threshold(self):
        config = self.make_config(mdeepcorr_stage1_threshold=1.0)
        with self.assertRaisesRegex(ValueError, "mdeepcorr_stage1_threshold"):
            config.validate()

    def test_mdeepcorr_requires_700_packet_data(self):
        config = self.make_config(
            target_model="mDeepcorr",
            data="mDeepcorr",
            flow_size=300,
            mdeepcorr100_model_path="missing-100.pth",
            mdeepcorr700_model_path="missing-700.pth",
        )
        with self.assertRaisesRegex(ValueError, "flow_size=700"):
            config.validate()

    def test_mdeepcorr_requires_explicit_stage_weights(self):
        config = self.make_config(
            target_model="mDeepcorr",
            data="mDeepcorr",
            flow_size=700,
        )
        with self.assertRaisesRegex(ValueError, "explicit mdeepcorr100_model_path"):
            config.validate()

    def test_mdeepcorr_config_uses_dedicated_deepcorr700_checkpoint_root(self):
        config_path = Path(__file__).parents[1] / "configs" / "mdeepcorr_config.jsonc"
        source = config_path.read_text(encoding="utf-8")
        self.assertIn('"checkpoints": "vista_augur/outputs/checkpoint_deepcorr700"', source)

    def test_deepcorr_config_uses_dedicated_deepcorr300_checkpoint_root(self):
        config_path = Path(__file__).parents[1] / "configs" / "deepcorr_config.jsonc"
        source = config_path.read_text(encoding="utf-8")
        self.assertIn('"checkpoints": "vista_augur/outputs/checkpoint_deepcorr300"', source)

    def test_deepcoffea_config_uses_work1_session_protocol(self):
        config_path = Path(__file__).parents[1] / "configs" / "deepcoffea_config.jsonc"
        source = config_path.read_text(encoding="utf-8")
        self.assertIn('"checkpoints": "vista_augur/outputs/checkpoint_deepcoffea"', source)
        self.assertIn('"deepcoffea_n_windows": 11', source)
        self.assertIn('"deepcoffea_vote_threshold": 9', source)

    def test_loads_commented_json_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.jsonc"
            path.write_text(
                """
                {
                  // 运行模式注释
                  "run_mode": "train",
                  "visible_gpu_devices": "0,1",
                  "backbone_model_path": "",
                  "backbone_model_name": "https://example.invalid/model",
                  "target_model_path": ""
                }
                """,
                encoding="utf-8",
            )
            config = load_config(path)

        self.assertEqual(config.run_mode, "train")
        self.assertEqual(config.visible_gpu_devices, "0,1")


if __name__ == "__main__":
    unittest.main()
