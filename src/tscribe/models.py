"""MLX Nemotron diarization and Whisper word transcription."""

from typing import TYPE_CHECKING, Literal, TypedDict, cast

from tscribe.audio import SAMPLE_RATE, Audio
from tscribe.transcript import SpeakerTurn, Word

if TYPE_CHECKING:
    from mlx_audio.vad.models.nemotron_diarization.nemotron_diarization import Model

DIARIZATION_MODEL = "mlx-community/Nemotron-3-Diarization-8bit"
DEFAULT_ASR_MODEL = "openai/whisper-base"
type Device = Literal["auto", "cpu", "cuda", "mps"]


class WordChunk(TypedDict):
    timestamp: tuple[float | None, float | None]
    text: str


class ASRResult(TypedDict):
    text: str
    chunks: list[WordChunk]


def resolve_device(requested: Device) -> str:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or --device auto.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use --device cpu or --device auto.")
    return requested


def diarize(audio: Audio, device: str) -> list[SpeakerTurn]:
    import mlx.core as mx
    from mlx_audio.vad import load

    mlx_device = mx.cpu if device == "cpu" else mx.gpu
    if not mx.is_available(mlx_device):
        raise ValueError("The requested MLX device is unavailable. Try --device cpu.")
    backend = "MLX CPU" if device == "cpu" else "MLX GPU"
    print(f"Identifying speakers with {DIARIZATION_MODEL} on {backend}...", flush=True)
    # Scope loading and inference to the chosen MLX device without changing global defaults.
    with mx.stream(mlx_device):
        model = cast("Model", load(DIARIZATION_MODEL, strict=True))
        result = model.generate(audio, sample_rate=SAMPLE_RATE)
    return [
        SpeakerTurn(segment.start, segment.end, f"Speaker {segment.speaker + 1}")
        for segment in result.segments
    ]


def transcribe(
    audio: Audio, device: str, model_id: str = DEFAULT_ASR_MODEL, language: str | None = None
) -> list[Word]:
    from transformers import pipeline

    print(f"Transcribing with {model_id} on {device}...", flush=True)
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        model_kwargs={"attn_implementation": "eager"},
    )
    options = {"task": "transcribe"}
    if language:
        options["language"] = language
    # Whisper's sequential long-form decoding handles recordings longer than 30s.
    result = cast(
        ASRResult,
        recognizer(
            {"raw": audio, "sampling_rate": SAMPLE_RATE},
            return_timestamps="word",
            generate_kwargs=options,
        ),
    )
    chunks = result.get("chunks", [])
    if not chunks and result["text"].strip():
        return [Word(None, None, result["text"])]
    return [Word(*chunk["timestamp"], chunk["text"]) for chunk in chunks]
