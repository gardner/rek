"""Optional MLX integration contracts, without Apple hardware or model downloads."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rek.cli import parse_args
from rek.transcript import SpeakerTurn


def test_mlx_is_default() -> None:
    assert parse_args([]).diarization_backend == "mlx"
    assert parse_args(["--diarization-backend", "mlx"]).diarization_backend == "mlx"


def test_mlx_uses_pinned_weights_and_shared_segment_policy() -> None:
    from rek.mlx_diarization import MODEL_REVISION, load_diarizer

    model = MagicMock()
    probs = np.zeros((5, 8), dtype=np.float32)
    probs[:3, 0] = 0.9
    probs[2:, 2] = 0.9
    model.generate.return_value.speaker_probs = probs
    vad = MagicMock()
    vad.load.return_value = model
    with (
        patch("rek.mlx_diarization.validate_platform"),
        patch.dict(
            "sys.modules", {"mlx_audio.vad": vad, "mlx": MagicMock(), "mlx.core": MagicMock()}
        ),
    ):
        infer = load_diarizer("cpu")
        turns = infer(np.zeros(752, dtype=np.float32))
        assert infer(np.zeros(0, dtype=np.float32)) == []
    assert turns == [SpeakerTurn(0, 0.03, "Speaker 1"), SpeakerTurn(0.02, 0.047, "Speaker 3")]
    assert vad.load.call_args.kwargs == {"strict": True, "revision": MODEL_REVISION}
    model.set_streaming_config.assert_called_once_with("offline")


def test_unsupported_platform_fails_before_import() -> None:
    from rek.mlx_diarization import validate_platform

    with patch("rek.mlx_diarization.sys.platform", "linux"):
        with pytest.raises(ValueError, match="Apple Silicon"):
            validate_platform()


def test_missing_extra_has_install_instructions() -> None:
    from rek.mlx_diarization import load_diarizer

    with (
        patch("rek.mlx_diarization.validate_platform"),
        patch.dict("sys.modules", {"mlx_audio.vad": None}),
        pytest.raises(RuntimeError, match="uv sync"),
    ):
        load_diarizer("coreml")
