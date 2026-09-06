from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from run_work2 import GpuState, _select_gpus


class LauncherTests(unittest.TestCase):
    def make_config(self, **overrides):
        values = {
            "visible_gpu_devices": "auto",
            "required_gpu_count": 2,
            "gpu_max_memory_used_mb": 1024,
            "gpu_max_utilization_percent": 10,
            "gpu_min_free_memory_mb": 14000,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @patch("run_work2._query_compute_processes", return_value={})
    @patch("run_work2._query_gpus")
    def test_auto_selects_exactly_two_idle_gpus(self, query_gpus, _query_processes):
        query_gpus.return_value = [
            GpuState("0", "GPU-0", "A", 24576, 100, 0),
            GpuState("1", "GPU-1", "B", 24576, 200, 1),
            GpuState("2", "GPU-2", "C", 24576, 8000, 90),
        ]

        self.assertEqual(_select_gpus(self.make_config()), "0,1")

    @patch("run_work2._query_compute_processes", return_value={"GPU-1": ["pid=42"]})
    @patch("run_work2._query_gpus")
    def test_rejects_explicit_gpu_with_existing_process(self, query_gpus, _query_processes):
        query_gpus.return_value = [
            GpuState("0", "GPU-0", "A", 24576, 100, 0),
            GpuState("1", "GPU-1", "B", 24576, 200, 1),
        ]

        with self.assertRaisesRegex(RuntimeError, "GPU"):
            _select_gpus(self.make_config(visible_gpu_devices="0,1"))


if __name__ == "__main__":
    unittest.main()
