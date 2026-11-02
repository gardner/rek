"""Nemotron cache and segment contracts; no model weights or native inference."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tscribe.diarization import diarize, speaker_turns
from tscribe.speaker_cache import SpeakerCache
from tscribe.transcript import SpeakerTurn


def test_overlapping_speakers_and_final_segment_are_preserved() -> None:
    logits = np.full((5, 8), -1, dtype=np.float32)
    logits[:3, 0] = 1
    logits[2:, 2] = 1
    assert speaker_turns(logits, 0.047) == [
        SpeakerTurn(0.0, 0.03, "Speaker 1"),
        SpeakerTurn(0.02, 0.047, "Speaker 3"),
    ]


def test_silent_frames_do_not_create_speakers() -> None:
    assert speaker_turns(np.zeros((10, 8), dtype=np.float32), 0.1) == []


def test_cache_stays_bounded_and_keeps_recent_frames() -> None:
    cache = SpeakerCache()
    for index in range(4):
        chunk = np.full((340, 512), index + 1, dtype=np.float32)
        logits = np.zeros(((len(cache.embeds) + 340) * 8, 8), dtype=np.float32)
        logits[:, index] = 2
        cache.update(chunk, logits, np.zeros(512, dtype=np.float32))
        assert len(cache.embeds) <= 304
        np.testing.assert_array_equal(cache.fifo, chunk[-40:])
    assert cache.compressed


@pytest.mark.parametrize("device", ["cpu", "coreml"])
def test_onnx_provider_and_external_weights(device: str) -> None:
    backend = MagicMock()
    backend.get_available_providers.return_value = [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    session = backend.InferenceSession.return_value
    session.run.return_value = [
        np.ones((1, 8, 8), dtype=np.float32),
        np.zeros((1, 1, 512), dtype=np.float32),
        np.zeros(512, dtype=np.float32),
    ]
    with (
        patch.dict("sys.modules", {"onnxruntime": backend}),
        patch("tscribe.diarization.model_file", side_effect=lambda _, name: name) as download,
        patch(
            "tscribe.diarization.mel_features", return_value=np.zeros((5, 128), dtype=np.float32)
        ),
    ):
        turns = diarize(np.zeros(800, dtype=np.float32), device)
    assert len(turns) == 8
    assert turns[0] == SpeakerTurn(0.0, 0.05, "Speaker 1")
    assert any(call.args[1] == "onnx/model.onnx_data" for call in download.call_args_list)
    providers = backend.InferenceSession.call_args.kwargs["providers"]
    assert providers[-1] == "CPUExecutionProvider"
    assert (len(providers) == 2) == (device == "coreml")


def test_chunk_cache_and_lookahead_do_not_duplicate_output() -> None:
    backend = MagicMock()
    session = backend.InferenceSession.return_value
    # 340 core encoder frames and 10 look-ahead on the first pass; the second
    # sees 264 speaker-cache frames + 40 FIFO frames followed by 10 final frames.
    session.run.side_effect = [
        [
            np.ones((1, 2800, 8), dtype=np.float32),
            np.ones((1, 350, 512), dtype=np.float32),
            np.zeros(512, dtype=np.float32),
        ],
        [
            np.ones((1, 2512, 8), dtype=np.float32),
            np.ones((1, 10, 512), dtype=np.float32),
            np.zeros(512, dtype=np.float32),
        ],
    ]
    with (
        patch.dict("sys.modules", {"onnxruntime": backend}),
        patch("tscribe.diarization.model_file", return_value="model.onnx"),
        patch(
            "tscribe.diarization.mel_features", return_value=np.zeros((2800, 128), dtype=np.float32)
        ),
    ):
        turns = diarize(np.zeros(28 * 16_000, dtype=np.float32), "cpu")
    assert turns[0] == SpeakerTurn(0.0, 28.0, "Speaker 1")
    second_inputs = session.run.call_args_list[1].args[1]
    assert second_inputs["cached_embeds"].shape == (1, 304, 512)
    assert second_inputs["input_features"].shape == (1, 80, 128)
    assert second_inputs["attention_mask"].shape == (1, 314)
