"""Timing boundaries and worker failures should not misrepresent comparisons."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rek.benchmark import compare, measure
from rek.transcript import SpeakerTurn


def test_model_reused_and_first_pass_excluded_from_warm_median() -> None:
    infer = MagicMock(return_value=[SpeakerTurn(0, 1, "Speaker 1")])
    with (
        patch("rek.benchmark.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
        patch("rek.diarization.load_diarizer", return_value=infer) as load,
        patch("rek.benchmark.perf_counter", side_effect=[0, 2, 2, 7, 7, 8, 8, 11]),
    ):
        result = measure(Path("audio.wav"), "onnx", "cpu", 2)
    load.assert_called_once_with("cpu")
    assert infer.call_count == 3
    assert result["load_seconds_including_uncached_downloads"] == 2
    assert result["first_inference_seconds"] == 5
    assert result["warm_median_seconds"] == 2
    assert result["warm_real_time_factor"] == 2


def test_worker_failure_is_saved_in_report() -> None:
    with (
        patch("rek.benchmark.sys.platform", "linux"),
        patch(
            "rek.benchmark.subprocess.run", return_value=MagicMock(returncode=1, stderr="failed")
        ),
    ):
        result = compare(Path("audio.wav"), 1, 30)
    assert result["results"] == [{"backend": "onnx", "requested_device": "cpu", "error": "failed"}]


def test_empty_audio_is_rejected_before_loading_model() -> None:
    with (
        patch("rek.benchmark.load_audio", return_value=np.zeros(0, dtype=np.float32)),
        pytest.raises(ValueError, match="10 ms"),
    ):
        measure(Path("audio.wav"), "onnx", "cpu", 1)
