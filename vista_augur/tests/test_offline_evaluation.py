from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.artifacts import correlation_conditions, load_score_conditions
from data.metrics import confusion_at_threshold, evaluate_scores
from data.overhead import evaluate_overhead


class OfflineEvaluationTests(unittest.TestCase):
    def test_low_fpr_uses_work1_rank_semantics(self):
        summary, curves = evaluate_scores([0.9, 0.7], [0.8, 0.2, 0.1], [1e-3])
        point = summary["operating_points"]["fpr_1e-03"]
        self.assertAlmostEqual(point["threshold"], 0.8)
        self.assertAlmostEqual(point["actual_fpr"], 1.0 / 3.0)
        self.assertFalse(point["resolvable"])
        self.assertAlmostEqual(summary["roc_auc"], 5.0 / 6.0)
        self.assertEqual(curves["fpr"][0], 0.0)

    def test_fixed_threshold_confusion(self):
        result = confusion_at_threshold([0.9, 0.4], [0.8, 0.1], 0.5)
        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(result["f1"], 0.5)

    def test_work2_npz_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            np.savez_compressed(
                path,
                clean_positive_scores=[0.9],
                clean_negative_scores=[0.1],
                adv_positive_scores=[0.4],
                adv_negative_scores=[0.2],
            )
            conditions = load_score_conditions(path)
        self.assertEqual(set(conditions), {"clean", "adv"})

    def test_overhead_reports_physical_violations(self):
        original = np.ones((1, 2, 3), dtype=np.float64)
        defended = original.copy()
        defended[:, 0, :] += 0.5
        defended[:, 1, :] += 0.25
        report = evaluate_overhead(original, defended, time_channels=[0], size_channels=[1])
        self.assertEqual(report["physical_constraints"]["time_negative_delta_count"], 0)
        self.assertAlmostEqual(report["aggregate"]["relative_delay_overhead"], 0.5)
        self.assertAlmostEqual(report["aggregate"]["relative_size_overhead"], 0.25)

    def test_deepcoffea_vote_is_kth_largest_window_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.npz"
            first = np.array([[0.9, 0.4], [0.3, 0.8]])
            second = np.array([[0.7, 0.5], [0.2, 0.6]])
            block = np.block([[first, np.zeros((2, 2))], [np.zeros((2, 2)), second]])
            np.savez_compressed(path, corr_matrix=block)
            conditions = correlation_conditions(path, n_windows=2, vote_threshold=2)
        positive, negative = conditions["corr"]
        np.testing.assert_allclose(positive, [0.7, 0.6])
        np.testing.assert_allclose(np.sort(negative), [0.2, 0.4])


if __name__ == "__main__":
    unittest.main()
