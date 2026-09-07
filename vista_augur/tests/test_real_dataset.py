from __future__ import annotations

import tempfile
import unittest
import pickle
from pathlib import Path

import numpy as np

from second_workpoint.config import ExperimentConfig
from second_workpoint.data.factory import build_dataset
from second_workpoint.data.real_dataset import (
    DEEP_CORR_RUN_NAMES,
    DeepCoffeaRealDataset,
    DeepCorrRealDataset,
    MDeepCorrRealDataset,
    _build_deterministic_split,
    _build_negative_index_map,
)


class DeepCoffeaDatasetTests(unittest.TestCase):
    def test_returns_one_session_with_work1_partitioned_target_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = "fixture"
            session_path = Path(directory) / f"{prefix}_test_session.npz"
            window_path = Path(directory) / f"{prefix}_test.npz"
            values = np.asarray(
                [
                    np.arange(10, dtype=np.float32),
                    np.arange(10, dtype=np.float32) + 100,
                ],
                dtype=object,
            )
            np.savez(
                session_path,
                tor_ipds=values,
                tor_sizes=values + 1,
                exit_ipds=values + 2,
                exit_sizes=values + 3,
                labels=np.asarray(["paired-flow", "negative-flow"]),
            )
            test_tor = np.arange(2 * 2 * 16, dtype=np.float32).reshape(2, 2, 16)
            test_exit = np.arange(2 * 2 * 18, dtype=np.float32).reshape(2, 2, 18) + 1000
            np.savez(
                window_path,
                test_tor=test_tor,
                test_exit=test_exit,
                test_label=np.asarray([["paired-flow", "negative-flow"]] * 2),
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
                deepcoffea_n_windows=2,
                deepcoffea_vote_threshold=2,
                backbone_model_path="",
            )
            dataset = DeepCoffeaRealDataset(config, split="eval")
            sample = dataset[0]
            self.assertEqual(len(dataset), 2)
            self.assertEqual(sample["full_flow"].shape, (2, 10))
            self.assertEqual(sample["history_seq"].shape, (3, 2, 4))
            self.assertEqual(sample["clean_future"].shape, (3, 2, 2))
            self.assertEqual(sample["target_tor_windows"].shape, (2, 16))
            self.assertEqual(sample["target_exit_windows"].shape, (2, 18))
            self.assertEqual(sample["target_negative_exit_windows"].shape, (1, 2, 18))
            np.testing.assert_allclose(sample["target_tor_windows"], test_tor[:, 0])
            np.testing.assert_allclose(sample["target_exit_windows"], test_exit[:, 0])
            np.testing.assert_allclose(
                sample["target_negative_exit_windows"][0],
                test_exit[:, 1],
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
                test_samples=2,
                deepcorr_unused_samples=1,
                eval_split="test",
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

    def test_mdeepcorr_uses_raw_300_named_files_at_700_packet_length(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_sample = {
                "here": [
                    {"<-": list(np.arange(700, dtype=np.float32) / 1000.0), "->": [0.2] * 700},
                    {"<-": [300.0] * 700, "->": [400.0] * 700},
                ],
                "there": [
                    {"->": [0.3] * 700, "<-": [0.4] * 700},
                    {"->": [500.0] * 700, "<-": [600.0] * 700},
                ],
            }
            for run_name in DEEP_CORR_RUN_NAMES:
                with (Path(directory) / f"{run_name}_tordata300.pickle").open("wb") as file:
                    pickle.dump([raw_sample], file)

            config = ExperimentConfig(
                data="mDeepcorr",
                target_model="mDeepcorr",
                data_path=directory,
                use_cached_indices=False,
                val_samples=1,
                test_samples=2,
                deepcorr_unused_samples=1,
                eval_split="test",
                flow_size=700,
                seq_len=96,
                pred_len=48,
                patch_len=4,
                stride=48,
                enc_in=4,
                backbone_model_path="",
            )
            sample = MDeepCorrRealDataset(config, split="eval")[0]

            self.assertIsInstance(build_dataset(config, split="eval"), MDeepCorrRealDataset)
            self.assertEqual(sample["history_seq"].shape, (12, 4, 96))
            self.assertEqual(sample["clean_future"].shape, (12, 48, 4))
            self.assertEqual(sample["writeback_meta"][0].tolist(), [96, 48])
            self.assertEqual(sample["writeback_meta"][-1].tolist(), [624, 48])
            self.assertEqual(sample["target_full_flow"].shape, (8, 700))
            self.assertGreater(float(sample["target_full_flow"][0, 699]), 0.0)

    def test_work1_split_reserves_unused_samples(self):
        values = {
            split: _build_deterministic_split(
                total_size=20,
                split=split,
                split_seed=2025,
                val_size=3,
                test_size=4,
                unused_size=5,
            )
            for split in ("train", "val", "test")
        }
        self.assertEqual(len(values["train"]), 8)
        self.assertEqual(len(values["val"]), 3)
        self.assertEqual(len(values["test"]), 4)
        self.assertTrue(set(values["train"]).isdisjoint(values["val"]))
        self.assertTrue(set(values["train"]).isdisjoint(values["test"]))
        self.assertTrue(set(values["val"]).isdisjoint(values["test"]))

    def test_work1_negative_pairs_are_deterministic_distinct_mismatches(self):
        indices = np.arange(1000, dtype=np.int64)
        first = _build_negative_index_map(indices, negative_count=199, split_seed=2025)
        second = _build_negative_index_map(indices, negative_count=199, split_seed=2025)

        self.assertEqual(first, second)
        for positive_index, negative_indices in first.items():
            self.assertEqual(len(negative_indices), 199)
            self.assertEqual(len(set(negative_indices)), 199)
            self.assertNotIn(positive_index, negative_indices)


if __name__ == "__main__":
    unittest.main()
