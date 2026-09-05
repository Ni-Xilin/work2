from __future__ import annotations

import unittest

import numpy as np

from second_workpoint.data.preprocessing import (
    build_deepcorr_full_flow,
    iter_window_starts,
)


class PreprocessingTests(unittest.TestCase):
    def test_deepcorr_channel_protocol(self):
        sample = {
            "here": [{"<-": [1.0], "->": [2.0]}, {"<-": [3000.0], "->": [4000.0]}],
            "there": [{"->": [5.0], "<-": [6.0]}, {"->": [7000.0], "<-": [8000.0]}],
        }
        flow = build_deepcorr_full_flow(sample, flow_size=2)
        np.testing.assert_allclose(flow[:, 0], [1000, 5000, 6000, 2000, 3, 7, 8, 4])
        np.testing.assert_allclose(flow[:, 1], 0.0)

    def test_tail_window_is_explicit(self):
        self.assertEqual(iter_window_starts(220, 150, 70, 70, include_tail=False), [0])
        self.assertEqual(iter_window_starts(221, 150, 70, 70, include_tail=True), [0, 70])

if __name__ == "__main__":
    unittest.main()
