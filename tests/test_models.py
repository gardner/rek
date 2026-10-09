"""Model routing and speaker conversion without loading weights or opening a GPU."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tscribe.models import Device, diarize, resolve_device, transcribe
from tscribe.transcript import SpeakerTurn, Word


@pytest.fixture
def mlx_core(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    core = MagicMock(cpu="cpu", gpu="gpu")
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    # Accidental imports of either former backend must not load native libraries.
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    return core


@pytest.mark.parametrize(
    ("requested", "gpu_available", "expected"),
    [
        ("auto", True, "gpu"),
        ("auto", False, "cpu"),
        ("cpu", True, "cpu"),
        ("mps", True, "gpu"),
        ("gpu", True, "gpu"),
    ],
)
def test_automatic_device_uses_available_acceleration(
    mlx_core: MagicMock, requested: Device, gpu_available: bool, expected: str
) -> None:
    mlx_core.is_available.return_value = gpu_available
    mlx_core.metal.is_available.return_value = gpu_available
    assert resolve_device(requested) == expected


def test_explicit_unavailable_gpu_is_an_error(mlx_core: MagicMock) -> None:
    mlx_core.is_available.return_value = False
    with pytest.raises(ValueError, match="unavailable"):
        resolve_device("gpu")


@pytest.mark.parametrize(("device", "stream_device"), [("gpu", "gpu"), ("cpu", "cpu")])
def test_mlx_diarization_preserves_speaker_ids_and_timestamps(
    mlx_core: MagicMock, device: str, stream_device: str
) -> None:
    vad = MagicMock()
    model = vad.load.return_value
    model.generate.return_value = SimpleNamespace(
        segments=[
            SimpleNamespace(start=0.0, end=1.5, speaker=0),
            SimpleNamespace(start=1.0, end=2.0, speaker=2),
            SimpleNamespace(start=3.0, end=4.0, speaker=0),
        ]
    )
    audio = np.zeros(64_000, dtype=np.float32)
    with patch.dict("sys.modules", {"mlx_audio.vad": vad}):
        turns = diarize(audio, device)

    vad.load.assert_called_once_with("mlx-community/Nemotron-3-Diarization-8bit", strict=True)
    mlx_core.stream.assert_called_once_with(stream_device)
    model.generate.assert_called_once()
    assert model.generate.call_args.args[0] is audio
    assert model.generate.call_args.kwargs["sample_rate"] == 16_000
    assert turns == [
        SpeakerTurn(0.0, 1.5, "Speaker 1"),
        SpeakerTurn(1.0, 2.0, "Speaker 3"),
        SpeakerTurn(3.0, 4.0, "Speaker 1"),
    ]


def test_parakeet_v2_returns_complete_timestamped_words(mlx_core: MagicMock) -> None:
    stt = MagicMock()
    model = stt.load.return_value
    model.preprocessor_config.sample_rate = 16_000
    model.generate.return_value = SimpleNamespace(
        text="Hello, everyone.",
        sentences=[
            SimpleNamespace(
                tokens=[
                    SimpleNamespace(start=0.0, end=0.1, text=" Hel"),
                    SimpleNamespace(start=0.1, end=0.3, text="lo,"),
                    SimpleNamespace(start=0.4, end=0.6, text=" every"),
                    SimpleNamespace(start=0.6, end=0.8, text="one."),
                ]
            )
        ],
    )
    audio = np.zeros(16_000, dtype=np.float32)
    with patch.dict("sys.modules", {"mlx_audio.stt": stt}):
        words = transcribe(audio, "gpu")

    stt.load.assert_called_once_with("mlx-community/parakeet-tdt-0.6b-v2", strict=True)
    mlx_core.stream.assert_called_once_with("gpu")
    model.generate.assert_called_once_with(
        mlx_core.array.return_value,
        dtype=mlx_core.float32,
        chunk_duration=30.0,
        overlap_duration=2.0,
    )
    assert words == [Word(0.0, 0.3, " Hello,"), Word(0.4, 0.8, " everyone.")]


def test_parakeet_rejects_non_english_before_loading(mlx_core: MagicMock) -> None:
    stt = MagicMock()
    with patch.dict("sys.modules", {"mlx_audio.stt": stt}):
        with pytest.raises(ValueError, match="English"):
            transcribe(np.zeros(16_000, dtype=np.float32), "gpu", language="fr")
    stt.load.assert_not_called()


@pytest.mark.parametrize(("text", "expected"), [("", []), ("Hi", [Word(None, None, "Hi")])])
def test_parakeet_preserves_text_without_timestamps(
    mlx_core: MagicMock, text: str, expected: list[Word]
) -> None:
    stt = MagicMock()
    stt.load.return_value.preprocessor_config.sample_rate = 16_000
    stt.load.return_value.generate.return_value = SimpleNamespace(text=text, sentences=[])
    with patch.dict("sys.modules", {"mlx_audio.stt": stt}):
        assert transcribe(np.zeros(16_000, dtype=np.float32), "cpu") == expected
