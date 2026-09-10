"""Tests for evaluation-only GPU isolation controls."""

from __future__ import annotations

from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from eval.common import _assert_gpus_idle, _parse_gpu_pair


def test_idle_gpu_is_accepted():
    result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("eval.common.subprocess.run", return_value=result):
        _assert_gpus_idle((2, 3), allow_busy=False)


def test_busy_gpu_is_rejected_in_strict_mode():
    result = CompletedProcess(args=[], returncode=0, stdout="1234, python, 4096\n", stderr="")
    with patch("eval.common.subprocess.run", return_value=result):
        with pytest.raises(RuntimeError, match="already have compute processes"):
            _assert_gpus_idle((2, 3), allow_busy=False)


def test_busy_gpu_is_allowed_in_operator_decision_mode():
    result = CompletedProcess(args=[], returncode=0, stdout="1234, python, 4096\n", stderr="")
    with patch("eval.common.subprocess.run", return_value=result):
        _assert_gpus_idle((2, 3), allow_busy=True)


def test_gpu_pair_maps_backbone_then_generator_target():
    assert _parse_gpu_pair("2,3") == (2, 3)
    with pytest.raises(ValueError, match="distinct"):
        _parse_gpu_pair("3,3")
