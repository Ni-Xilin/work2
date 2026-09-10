"""Regression tests for the formal DeepCoFFEA evaluation representation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from data.artifacts import _session_matrix
from eval.deepcoffea.evaluate_deepcoffea import PROFILE as DEEPCOFFEA_PROFILE
from eval.deepcorr300.evaluate_deepcorr300 import PROFILE as DEEPCORR_PROFILE
from eval.mdeepcorr.evaluate_mdeepcorr import PROFILE as MDEEPCORR_PROFILE
from eval.common import (
    EvaluationProfile,
    _mean_relative_l2,
    _save_deepcoffea_work1_outputs,
    _save_deepcorr_work1_samples,
    _save_deepcorr_work1_scores,
    _window_matrices,
    _work1_interleaved_scores,
    _work1_mdeepcorr_negative_pair_indices,
    _work1_negative_pair_indices,
    _work1_positive_first_scores,
)
from second_workpoint.training.deepcoffea_protocol import hard_partition_sessions_by_ipd


def test_evaluator_profiles_match_work1_batch_and_output_defaults():
    assert (DEEPCORR_PROFILE.batch_size, DEEPCORR_PROFILE.num_workers) == (16, 0)
    assert (DEEPCORR_PROFILE.shuffle, DEEPCORR_PROFILE.drop_last) == (False, True)
    assert DEEPCORR_PROFILE.scoring_batch_size == 256
    assert DEEPCORR_PROFILE.model_dir == "eval/deepcorr300"

    assert (MDEEPCORR_PROFILE.batch_size, MDEEPCORR_PROFILE.num_workers) == (16, 32)
    assert (MDEEPCORR_PROFILE.shuffle, MDEEPCORR_PROFILE.drop_last) == (False, True)
    assert MDEEPCORR_PROFILE.scoring_batch_size == 64
    assert MDEEPCORR_PROFILE.model_dir == "eval/mdeepcorr"

    assert (DEEPCOFFEA_PROFILE.batch_size, DEEPCOFFEA_PROFILE.num_workers) == (25, 0)
    assert (DEEPCOFFEA_PROFILE.shuffle, DEEPCOFFEA_PROFILE.drop_last) == (True, True)
    assert (DEEPCOFFEA_PROFILE.max_steps, DEEPCOFFEA_PROFILE.scoring_batch_size) == (20, 275)
    assert DEEPCOFFEA_PROFILE.model_dir == "eval/deepcoffea"


def test_hard_partition_uses_compact_work1_layout():
    torch = pytest.importorskip("torch")
    session = torch.tensor(
        [
            [1000.0, 1000.0, 1000.0, 1000.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )

    windows = hard_partition_sessions_by_ipd(
        [session],
        delta_seconds=1.0,
        window_seconds=2.0,
        window_count=1,
        packet_limit=4,
    )

    # searchsorted(left) selects [packet 0], then Work1 packs IPD || size
    # before padding. The size therefore immediately follows the single IPD.
    np.testing.assert_allclose(windows.numpy()[0, 0], [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_window_matrix_layout_and_vote_reduction():
    tor = np.asarray([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
    exits = tor.copy()
    matrices = _window_matrices(tor, exits)

    assert matrices.shape == (2, 2, 2)
    np.testing.assert_array_equal(matrices[0], np.eye(2, dtype=np.float32))
    np.testing.assert_array_equal(matrices[1], np.eye(2, dtype=np.float32))
    np.testing.assert_array_equal(_session_matrix(matrices, n_windows=2, vote_threshold=2), np.eye(2))


def test_work1_negative_pairs_use_random_tor_and_fixed_exit():
    tor_indices, exit_indices = _work1_negative_pair_indices(4, 2, seed=2025)

    assert tor_indices.shape == exit_indices.shape == (8,)
    for exit_index in range(4):
        group = slice(exit_index * 2, (exit_index + 1) * 2)
        np.testing.assert_array_equal(exit_indices[group], [exit_index, exit_index])
        assert exit_index not in tor_indices[group]
        assert len(np.unique(tor_indices[group])) == 2


def test_work1_score_artifact_orders_negatives_before_each_positive():
    outputs, labels = _work1_interleaved_scores([0.9, 0.8], [0.1, 0.2, 0.3, 0.4], 2)

    np.testing.assert_allclose(outputs, [0.1, 0.2, 0.9, 0.3, 0.4, 0.8])
    np.testing.assert_array_equal(labels, [0, 0, 1, 0, 0, 1])


def test_mdeepcorr_orders_positive_before_per_entry_negatives():
    outputs, labels = _work1_positive_first_scores([0.9, 0.8], [0.1, 0.2, 0.3, 0.4], 2)

    np.testing.assert_allclose(outputs, [0.9, 0.1, 0.2, 0.8, 0.3, 0.4])
    np.testing.assert_array_equal(labels, [1, 0, 0, 1, 0, 0])
    tor_indices, exit_indices = _work1_mdeepcorr_negative_pair_indices(4, 2)
    np.testing.assert_array_equal(tor_indices, [0, 0, 1, 1, 2, 2, 3, 3])
    for tor_index in range(4):
        group = slice(tor_index * 2, (tor_index + 1) * 2)
        assert tor_index not in exit_indices[group]


def test_work1_output_names_and_schemas_are_written_under_requested_eval_dir(tmp_path):
    profile = EvaluationProfile(
        name="deepcorr300",
        config="unused",
        checkpoint="auto",
        model_dir="eval/deepcorr300",
        target_names=("deepcorr300",),
        protocol="deepcorr_1_to_199",
        batch_size=16,
        num_workers=0,
        shuffle=False,
        drop_last=True,
    )
    sequences = {
        "adv_samples": np.ones((1, 1, 8, 2), dtype=np.float32),
        "original_samples": np.zeros((1, 1, 8, 2), dtype=np.float32),
        "time_ratios": np.asarray(0.125),
        "size_ratios": np.asarray(0.25),
    }
    numpy_tensor_stub = SimpleNamespace(from_numpy=lambda value: value)
    sample_path = _save_deepcorr_work1_samples(
        tmp_path, sequences, profile, torch_module=numpy_tensor_stub
    )
    score_path = _save_deepcorr_work1_scores(
        tmp_path,
        {"adv_all_outputs": np.asarray([0.2]), "all_labels": np.asarray([0])},
        profile,
    )
    assert sample_path.name == "Gdeepcorr_advsamples_time0.1250_size0.2500.p"
    assert score_path.name == "Gtest_index300_time0_result.p"

    matrices = {
        "clean_work1_matrix": np.eye(2, dtype=np.float32),
        "adv_work1_matrix": np.eye(2, dtype=np.float32) * 0.5,
    }
    clean_path, adv_path = _save_deepcoffea_work1_outputs(tmp_path, sequences, matrices)
    assert clean_path.name == "corrmatrix_sim1.0000.npz"
    assert adv_path.name == "Gcorrmatrix_time0.1250_size0.2500_sim0.5000.npz"
    with np.load(adv_path) as payload:
        assert set(payload.files) == {"corr_matrix", "meantimeL2_rate", "meansizeL2_rate", "sim"}


def test_intermediate_relative_l2_is_mean_per_session():
    original = np.ones((2, 1, 2, 2), dtype=np.float32)
    adversarial = original.copy()
    adversarial[0, 0, 0] += 1.0
    mask = np.ones((2, 1, 2), dtype=np.uint8)

    expected = (np.sqrt(2.0) / np.sqrt(2.0) + 0.0) / 2.0
    assert _mean_relative_l2(original, adversarial, [0], mask) == pytest.approx(expected)


def test_relative_l2_matches_work1_channelwise_mean():
    original = np.asarray([[[[1.0, 0.0], [10.0, 0.0]]]], dtype=np.float32)
    adversarial = np.asarray([[[[2.0, 0.0], [12.0, 0.0]]]], dtype=np.float32)
    mask = np.ones((1, 1, 2), dtype=np.uint8)

    assert _mean_relative_l2(original, adversarial, [0, 1], mask) == pytest.approx((1.0 + 0.2) / 2.0)
