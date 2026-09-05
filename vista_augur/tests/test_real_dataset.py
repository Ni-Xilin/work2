from __future__ import annotations

import tempfile
import unittest
import pickle
from pathlib import Path

import numpy as np

from second_workpoint.config import ExperimentConfig
from second_workpoint.data.real_dataset import DEEP_CORR_RUN_NAMES, DeepCoffeaRealDataset, DeepCorrRealDataset


class DeepCoffeaDatasetTests(unittest.TestCase):
    def test_returns_the_real_paired_exit_window(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = "fixture"
            path = Path(directory) / f"{prefix}_test_session.npz"
            values = np.asarray(
                [
                    np.arange(10, dtype=np.float32),
                    np.arange(10, dtype=np.float32) + 100,
                ],
                dtype=object,
            )
            np.savez(
                path,
                tor_ipds=values,
                tor_sizes=values + 1,
                exit_ipds=values + 2,
                exit_sizes=values + 3,
                labels=np.asarray(["paired-flow", "negative-flow"]),
            )
            config = ExperimentConfig(
                data="DeepCoFFEA",
                data_path=directory,
                deepcoffea_prefix=prefix,
                eval_split="test",
                flow_size=10,
                seq_len=4,
                pred_len=2,
                patch_len=2,
                stride=2,
                enc_in=2,
                time_channel_indices=[0],
                size_channel_indices=[1],
                deepcoffea_include_tail=False,
                deepcoffea_tor_len=8,
                deepcoffea_exit_len=9,
                backbone_model_path="",
            )
            dataset = DeepCoffeaRealDataset(config, split="eval")
            sample = dataset[0]
            self.assertEqual(sample["target_full_flow"].shape, (2, 8))
            self.assertEqual(sample["target_exit_flow"].shape, (2, 9))
            self.assertEqual(sample["target_negative_exit_flow"].shape, (1, 2, 9))
            np.testing.assert_allclose(sample["target_full_flow"][0], np.arange(10, dtype=np.float32)[:8])
            np.testing.assert_allclose(sample["target_exit_flow"][0], np.arange(10, dtype=np.float32)[:9] + 2)
            np.testing.assert_allclose(
                sample["target_negative_exit_flow"][0, 0],
                np.arange(10, dtype=np.float32)[:9] + 102,
            )
            self.assertEqual(sample["label_text"], "paired-flow")


class DeepCorrDatasetTests(unittest.TestCase):
    def test_groups_all_windows_from_one_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_sample = {
                "here": [
                    {"<-": [0.1] * 300, "->": [0.2] * 300},
                    {"<-": [300.0] * 300, "->": [400.0] * 300},
                ],
                "there": [
                    {"->": [0.3] * 300, "<-": [0.4] * 300},
                    {"->": [500.0] * 300, "<-": [600.0] * 300},
                ],
            }
            for run_name in DEEP_CORR_RUN_NAMES:
                with (Path(directory) / f"{run_name}_tordata300.pickle").open("wb") as file:
                    pickle.dump([raw_sample], file)

            config = ExperimentConfig(
                data="Deepcorr300",
                data_path=directory,
                use_cached_indices=False,
                val_samples=1,
                test_samples=0,
                flow_size=300,
                seq_len=96,
                pred_len=48,
                patch_len=4,
                stride=48,
                enc_in=4,
                backbone_model_path="",
            )
            sample = DeepCorrRealDataset(config, split="eval")[0]

            self.assertEqual(sample["history_seq"].shape, (4, 4, 96))
            self.assertEqual(sample["clean_future"].shape, (4, 48, 4))
            self.assertEqual(sample["writeback_meta"].tolist(), [[96, 48], [144, 48], [192, 48], [240, 48]])
            self.assertEqual(sample["target_full_flow"].shape, (8, 300))
            self.assertEqual(sample["target_negative_flow"].shape, (1, 8, 300))


if __name__ == "__main__":
    unittest.main()
