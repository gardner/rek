"""Model routing and speaker conversion without loading weights or opening a GPU."""

from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tscribe.models import Device, diarize, resolve_device
from tscribe.transcript import SpeakerTurn


@pytest.mark.parametrize(
    ("requested", "cuda", "mps", "expected"),
    [
        ("auto", False, True, "mps"),
        ("auto", True, True, "cuda"),
        ("auto", False, False, "cpu"),
        ("cpu", False, True, "cpu"),
    ],
)
def test_automatic_device_uses_available_acceleration(
    requested: Device, cuda: bool, mps: bool, expected: str
) -> None:
    torch = MagicMock()
    torch.cuda.is_available.return_value = cuda
    torch.backends.mps.is_available.return_value = mps
    with patch.dict("sys.modules", {"torch": torch}):
        assert resolve_device(requested) == expected


@pytest.mark.parametrize(("device", "stream_device"), [("mps", "gpu"), ("cpu", "cpu")])
def test_mlx_diarization_preserves_speaker_ids_and_timestamps(
    device: str, stream_device: str
) -> None:
    core = MagicMock(cpu="cpu", gpu="gpu")
    mlx = ModuleType("mlx")
    mlx.core = core
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
    # Hide the old backend so an accidental Transformers load cannot download weights.
    with patch.dict(
        "sys.modules",
        {
            "mlx": mlx,
            "mlx.core": core,
            "mlx_audio.vad": vad,
            "transformers": ModuleType("transformers"),
            "torch": MagicMock(),
        },
    ):
        turns = diarize(audio, device)

    vad.load.assert_called_once_with("mlx-community/Nemotron-3-Diarization-8bit", strict=True)
    core.stream.assert_called_once_with(stream_device)
    model.generate.assert_called_once()
    assert model.generate.call_args.args[0] is audio
    assert model.generate.call_args.kwargs["sample_rate"] == 16_000
    assert turns == [
        SpeakerTurn(0.0, 1.5, "Speaker 1"),
        SpeakerTurn(1.0, 2.0, "Speaker 3"),
        SpeakerTurn(3.0, 4.0, "Speaker 1"),
    ]
