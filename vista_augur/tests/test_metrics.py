from __future__ import annotations

import unittest

from second_workpoint.training.trainer import TorchTrainer


class MetricTests(unittest.TestCase):
    def test_confusion_summary(self):
        totals = {"clean_tp": 8, "clean_fp": 2, "clean_tn": 18, "clean_fn": 2}
        summary = TorchTrainer._classification_summary(totals, "clean")
        self.assertTrue(summary["clean_classification_available"])
        self.assertAlmostEqual(summary["clean_precision"], 0.8)
        self.assertAlmostEqual(summary["clean_recall"], 0.8)
        self.assertAlmostEqual(summary["clean_f1"], 0.8)
        self.assertAlmostEqual(summary["clean_fpr"], 0.1)


if __name__ == "__main__":
    unittest.main()
