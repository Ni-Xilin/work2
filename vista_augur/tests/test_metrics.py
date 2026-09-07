from __future__ import annotations

import unittest

from second_workpoint.config import ExperimentConfig
from second_workpoint.training.trainer import TorchTrainer


class MetricTests(unittest.TestCase):
    def test_only_deepcoffea_uses_work1_training_monitor(self):
        trainer = TorchTrainer.__new__(TorchTrainer)
        trainer.config = ExperimentConfig(target_model="Deepcoffea", backbone_model_path="")
        self.assertTrue(trainer._uses_work1_deepcoffea_training_monitor())
        trainer.config = ExperimentConfig(target_model="Deepcorr300", backbone_model_path="")
        self.assertFalse(trainer._uses_work1_deepcoffea_training_monitor())

    def test_confusion_summary(self):
        totals = {"clean_tp": 8, "clean_fp": 2, "clean_tn": 18, "clean_fn": 2}
        summary = TorchTrainer._classification_summary(totals, "clean")
        self.assertTrue(summary["clean_classification_available"])
        self.assertAlmostEqual(summary["clean_precision"], 0.8)
        self.assertAlmostEqual(summary["clean_recall"], 0.8)
        self.assertAlmostEqual(summary["clean_f1"], 0.8)
        self.assertAlmostEqual(summary["clean_fpr"], 0.1)

    def test_checkpoint_name_contains_epoch_and_validation_metrics(self):
        stem = TorchTrainer._checkpoint_stem(
            7,
            {
                "original_positive_rate": 0.8126,
                "adv_positive_rate": 0.0744,
                "loss": 0.50426,
                "time_ratio": 0.1236,
                "size_ratio": 0.0984,
            },
        )
        self.assertEqual(
            stem,
            "generator_ep007_origrec0.813_advrec0.074_loss0.5043_time0.124_size0.098",
        )

    def test_deepcoffea_checkpoint_name_uses_work1_training_metrics(self):
        stem = TorchTrainer._deepcoffea_checkpoint_stem(
            7,
            {
                "mean_adv_logit": 0.1634,
                "loss": 0.50426,
                "time_ratio": 0.1236,
                "size_ratio": 0.0054,
            },
        )
        self.assertEqual(stem, "generator_ep007_cos0.163_loss0.5043_time0.124_size0.005")


if __name__ == "__main__":
    unittest.main()
