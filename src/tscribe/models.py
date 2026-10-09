"""MLX Nemotron diarization and English Parakeet v2 transcription."""

from typing import TYPE_CHECKING, Literal, cast

from tscribe.audio import SAMPLE_RATE, Audio
from tscribe.transcript import SpeakerTurn, Word, merge_word_tokens

if TYPE_CHECKING:
    from mlx_audio.stt.models.nemo.alignment import AlignedResult
    from mlx_audio.stt.models.parakeet.parakeet import Model as ParakeetModel
    from mlx_audio.vad.models.nemotron_diarization.nemotron_diarization import (
        Model as DiarizationModel,
    )

DIARIZATION_MODEL = "mlx-community/Nemotron-3-Diarization-8bit"
DEFAULT_ASR_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"
type Device = Literal["auto", "cpu", "gpu", "cuda", "mps"]
type InferenceDevice = Literal["cpu", "gpu"]


def resolve_device(requested: Device) -> InferenceDevice:
    import mlx.core as mx

    if requested == "auto":
        return "gpu" if mx.is_available(mx.gpu) else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "mps" and not mx.metal.is_available():
        raise ValueError("MLX Metal is unavailable. Use --device cpu or --device auto.")
    if requested == "cuda" and not mx.cuda.is_available():
        raise ValueError("MLX CUDA is unavailable. Use --device cpu or --device auto.")
    if not mx.is_available(mx.gpu):
        raise ValueError("The MLX GPU is unavailable. Use --device cpu.")
    return "gpu"


def diarize(audio: Audio, device: InferenceDevice) -> list[SpeakerTurn]:
    import mlx.core as mx
    from mlx_audio.vad import load

    mlx_device = mx.cpu if device == "cpu" else mx.gpu
    if not mx.is_available(mlx_device):
        raise ValueError("The requested MLX device is unavailable. Try --device cpu.")
    backend = "MLX CPU" if device == "cpu" else "MLX GPU"
    print(f"Identifying speakers with {DIARIZATION_MODEL} on {backend}...", flush=True)
    # Scope loading and inference to the chosen MLX device without changing global defaults.
    with mx.stream(mlx_device):
        model = cast("DiarizationModel", load(DIARIZATION_MODEL, strict=True))
        result = model.generate(audio, sample_rate=SAMPLE_RATE)
    return [
        SpeakerTurn(segment.start, segment.end, f"Speaker {segment.speaker + 1}")
        for segment in result.segments
    ]


def transcribe(
    audio: Audio,
    device: InferenceDevice,
    model_id: str = DEFAULT_ASR_MODEL,
    language: str | None = "en",
) -> list[Word]:
    if language is not None and language.lower() not in {"en", "english"}:
        raise ValueError("Parakeet v2 supports English only.")
    import mlx.core as mx
    from mlx_audio.stt import load

    print(f"Transcribing English with {model_id} on MLX {device.upper()}...", flush=True)
    with mx.stream(mx.cpu if device == "cpu" else mx.gpu):
        model = cast("ParakeetModel", load(model_id, strict=True))
        if model.preprocessor_config.sample_rate != SAMPLE_RATE:
            raise ValueError("The Parakeet model must accept 16 kHz audio.")
        # Bound attention memory and let MLX Audio merge overlapping chunk timestamps.
        result = cast(
            "AlignedResult",
            model.generate(
                mx.array(audio), dtype=mx.float32, chunk_duration=30.0, overlap_duration=2.0
            ),
        )
    words = merge_word_tokens(
        [
            Word(token.start, token.end, token.text)
            for sentence in result.sentences
            for token in sentence.tokens
        ]
    )
    if not words and result.text.strip():
        return [Word(None, None, result.text)]
    return words
