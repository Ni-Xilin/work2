from __future__ import annotations

import unittest

from second_workpoint.training.metrics import summarize_scores, threshold_at_target_fpr


class OperatingPointTests(unittest.TestCase):
    def test_unresolvable_target_reports_zero_empirical_fpr(self):
        result = threshold_at_target_fpr([0.1, 0.2, 0.3], 1e-4)
        self.assertFalse(result["resolvable"])
        self.assertEqual(result["actual_calibration_fpr"], 0.0)
        self.assertGreater(result["threshold"], 0.3)

    def test_score_summary_uses_supplied_threshold(self):
        result = summarize_scores([0.9, 0.4], [0.8, 0.1], threshold=0.5)
        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(result["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
