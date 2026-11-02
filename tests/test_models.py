"""ONNX model adapters, using mocks rather than downloading model weights."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gkit.models import resolve_device, transcribe
from gkit.transcript import Word


@pytest.mark.parametrize(
    ("platform", "requested", "expected"),
    [
        ("darwin", "auto", "coreml"),
        ("darwin", "mps", "coreml"),
        ("darwin", "gpu", "coreml"),
        ("darwin", "coreml", "coreml"),
        ("darwin", "cpu", "cpu"),
        ("linux", "auto", "cpu"),
        ("win32", "auto", "cpu"),
        ("win32", "cpu", "cpu"),
    ],
)
def test_selects_platform_provider(platform: str, requested: str, expected: str) -> None:
    with patch("gkit.models.sys.platform", platform):
        assert resolve_device(requested) == expected


@pytest.mark.parametrize("requested", ["gpu", "cuda", "mps", "coreml"])
def test_unavailable_acceleration_explains_cpu_option(requested: str) -> None:
    with patch("gkit.models.sys.platform", "linux"):
        with pytest.raises(ValueError, match="--device cpu"):
            resolve_device(requested)


@pytest.mark.parametrize("device", ["cpu", "coreml"])
def test_parakeet_merges_sentencepiece_tokens_with_durations(device: str) -> None:
    backend = MagicMock()
    recognizer = backend.OfflineRecognizer.from_transducer.return_value
    recognizer.create_stream.return_value.result = SimpleNamespace(
        text="Hello, everyone.",
        tokens=["▁Hel", "lo,", "▁every", "one."],
        timestamps=[0.0, 0.1, 0.4, 0.6],
        durations=[0.1, 0.2, 0.2, 0.2],
    )
    with (
        patch.dict("sys.modules", {"sherpa_onnx": backend}),
        patch("gkit.models.model_file", side_effect=lambda _, name: name),
    ):
        words = transcribe(np.zeros(16_000, dtype=np.float32), device)
    assert backend.OfflineRecognizer.from_transducer.call_args.kwargs["provider"] == device
    assert words == [Word(0.0, 0.30000000000000004, " Hello,"), Word(0.4, 0.8, " everyone.")]
    assert backend.OfflineRecognizer.from_transducer.call_args.kwargs["model_type"] == (
        "nemo_transducer"
    )


def test_parakeet_rejects_non_english_before_loading() -> None:
    with pytest.raises(ValueError, match="English"):
        transcribe(np.zeros(16_000, dtype=np.float32), "cpu", language="fr")


@pytest.mark.parametrize(("text", "expected"), [("", []), ("Hi", [Word(None, None, "Hi")])])
def test_preserves_text_without_timestamps(text: str, expected: list[Word]) -> None:
    backend = MagicMock()
    backend.OfflineRecognizer.from_transducer.return_value.create_stream.return_value.result = (
        SimpleNamespace(text=text, tokens=[], timestamps=[], durations=[])
    )
    with (
        patch.dict("sys.modules", {"sherpa_onnx": backend}),
        patch("gkit.models.model_file", return_value="model.onnx"),
    ):
        assert transcribe(np.zeros(16_000, dtype=np.float32), "cpu") == expected


def test_chunks_offset_times_and_discard_context_words() -> None:
    backend = MagicMock()
    recognizer = backend.OfflineRecognizer.from_transducer.return_value
    recognizer.create_stream.side_effect = [
        SimpleNamespace(
            accept_waveform=MagicMock(),
            result=SimpleNamespace(
                text="First", tokens=["▁First"], timestamps=[1.0], durations=[0.5]
            ),
        ),
        SimpleNamespace(
            accept_waveform=MagicMock(),
            result=SimpleNamespace(
                text="First Second",
                tokens=["▁First", "▁Second"],
                timestamps=[0.0, 3.0],
                durations=[0.5, 0.5],
            ),
        ),
    ]
    with (
        patch.dict("sys.modules", {"sherpa_onnx": backend}),
        patch("gkit.models.model_file", return_value="model.onnx"),
    ):
        words = transcribe(np.zeros(32 * 16_000, dtype=np.float32), "cpu")
    assert words == [Word(1.0, 1.5, " First"), Word(31.0, 31.5, " Second")]
