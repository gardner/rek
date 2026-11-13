"""MLX timestamps and chunk settings without loading model weights."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rek.mlx_asr import DEFAULT_ASR_MODEL, MODEL_REVISION, transcribe
from rek.transcript import Word


def test_aligned_tokens_merge_and_keep_absolute_timestamps() -> None:
    backend = MagicMock()
    core = MagicMock()
    backend.load_model.return_value.generate.return_value = SimpleNamespace(
        sentences=[
            SimpleNamespace(
                tokens=[
                    SimpleNamespace(text=" Hel", start=31.0, end=31.1),
                    SimpleNamespace(text="lo.", start=31.1, end=31.3),
                ]
            )
        ]
    )
    with (
        patch("rek.mlx_asr.validate_platform"),
        patch.dict(
            "sys.modules",
            {"mlx": MagicMock(core=core), "mlx.core": core, "mlx_audio.stt.utils": backend},
        ),
    ):
        words = transcribe(np.zeros(32 * 16000, dtype=np.float32), "coreml")
    assert words == [Word(31.0, 31.3, " Hello.")]
    backend.load_model.assert_called_once_with(
        DEFAULT_ASR_MODEL, strict=True, revision=MODEL_REVISION
    )
    assert backend.load_model.return_value.generate.call_args.kwargs == {
        "chunk_duration": 30.0,
        "overlap_duration": 2.0,
    }
    core.stream.assert_called_once_with(core.gpu)


def test_non_english_rejected_before_import() -> None:
    with pytest.raises(ValueError, match="English"):
        transcribe(np.zeros(16000, dtype=np.float32), "cpu", language="fr")


def test_missing_onnx_dependency_has_install_instructions() -> None:
    from rek.diarization import load_diarizer
    from rek.models import transcribe as onnx_transcribe

    with patch.dict("sys.modules", {"onnxruntime": None, "sherpa_onnx": None}):
        with pytest.raises(RuntimeError, match="uv sync --extra onnx"):
            load_diarizer("cpu")
        with pytest.raises(RuntimeError, match="uv sync --extra onnx"):
            onnx_transcribe(np.zeros(16000, dtype=np.float32), "cpu")
