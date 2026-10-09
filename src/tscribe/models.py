"""Local Nemotron diarization and Whisper word transcription."""

from typing import Literal, TypedDict, cast

from tscribe.audio import SAMPLE_RATE, Audio
from tscribe.transcript import SpeakerTurn, Word

DIARIZATION_MODEL = "nvidia/Nemotron-3-Diarization"
DEFAULT_ASR_MODEL = "openai/whisper-base"
type Device = Literal["auto", "cpu", "cuda", "mps"]


class SpeakerSegment(TypedDict):
    Start: float
    End: float
    Speaker: int


class WordChunk(TypedDict):
    timestamp: tuple[float | None, float | None]
    text: str


class ASRResult(TypedDict):
    text: str
    chunks: list[WordChunk]


def resolve_device(requested: Device) -> str:
    import torch

    if requested == "auto":
        # CPU is the portable default on Macs; MPS can be selected explicitly.
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or --device auto.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use --device cpu or --device auto.")
    return requested


def diarize(audio: Audio, device: str) -> list[SpeakerTurn]:
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    print(f"Identifying speakers with {DIARIZATION_MODEL} on {device}...", flush=True)
    processor = AutoProcessor.from_pretrained(DIARIZATION_MODEL)
    model = AutoModelForAudioFrameClassification.from_pretrained(DIARIZATION_MODEL)
    model.to(device).eval()
    inputs = processor(audio, sampling_rate=SAMPLE_RATE).to(device, dtype=model.dtype)
    with torch.inference_mode():
        logits = model(**inputs).logits
    segments = cast(
        list[SpeakerSegment], processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
    )
    return [
        SpeakerTurn(segment["Start"], segment["End"], f"Speaker {segment['Speaker'] + 1}")
        for segment in segments
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
