"""Optional Apple Silicon Nemotron backend with the same output policy as ONNX."""

import platform
import sys
from collections.abc import Callable
from typing import Literal, Protocol, cast

import numpy as np

from rek.audio import SAMPLE_RATE, Audio
from rek.diarization import speaker_turns
from rek.transcript import SpeakerTurn

MODEL_ID = "mlx-community/Nemotron-3-Diarization"
MODEL_REVISION = "59ed2dbfc1346dcea9d423c71306a3a2499c568f"


class DiarizationResult(Protocol):
    speaker_probs: object


class MLXModel(Protocol):
    def set_streaming_config(self, preset: str) -> None: ...

    def generate(self, audio: Audio, sample_rate: int = SAMPLE_RATE) -> DiarizationResult: ...


def validate_platform() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ValueError(
            "MLX diarization requires an Apple Silicon Mac; use --diarization-backend onnx."
        )


def load_diarizer(device: Literal["cpu", "coreml"]) -> Callable[[Audio], list[SpeakerTurn]]:
    """Load once; each inference starts with fresh speaker state for its recording."""
    validate_platform()
    try:
        # These dependencies are deliberately absent from the default installation.
        import mlx.core as mx  # pyright: ignore[reportMissingImports]
        from mlx_audio.vad import load  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("MLX dependencies are missing. Run: uv sync --extra mlx") from error
    target = mx.cpu if device == "cpu" else mx.gpu
    with mx.stream(target):
        model = cast(MLXModel, load(MODEL_ID, strict=True, revision=MODEL_REVISION))
        model.set_streaming_config("offline")

    def infer(audio: Audio) -> list[SpeakerTurn]:
        if len(audio) < 160:
            return []
        with mx.stream(target):
            result = model.generate(audio, sample_rate=SAMPLE_RATE)
            # Conversion synchronizes lazy MLX work before returning/timing it.
            probabilities = np.asarray(result.speaker_probs, dtype=np.float32)
        return speaker_turns(probabilities - np.float32(0.5), len(audio) / SAMPLE_RATE)

    return infer


def diarize(audio: Audio, device: Literal["cpu", "coreml"]) -> list[SpeakerTurn]:
    return load_diarizer(device)(audio)
