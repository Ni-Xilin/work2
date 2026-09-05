from __future__ import annotations

import unittest

from second_workpoint.config import ExperimentConfig


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


if __name__ == "__main__":
    unittest.main()
