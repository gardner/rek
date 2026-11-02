"""Portable ONNX speaker diarization and English Parakeet v2 transcription."""

import sys
from typing import Literal, Protocol

from tscribe.audio import SAMPLE_RATE, Audio
from tscribe.model_files import (
    DEFAULT_ASR_MODEL as DEFAULT_ASR_MODEL,
)
from tscribe.model_files import (
    DIARIZATION_MODEL as DIARIZATION_MODEL,
)
from tscribe.model_files import (
    EMBEDDING_FILE,
    EMBEDDING_MODEL,
    model_file,
)
from tscribe.transcript import SpeakerTurn, Word, merge_word_tokens

type Device = Literal["auto", "cpu", "gpu", "cuda", "mps", "coreml"]
type InferenceDevice = Literal["cpu", "coreml"]
_CHUNK_SAMPLES = 30 * SAMPLE_RATE
_CONTEXT_SAMPLES = 2 * SAMPLE_RATE


class RecognitionResult(Protocol):
    text: str
    tokens: list[str]
    timestamps: list[float]
    durations: list[float]


def resolve_device(requested: Device) -> InferenceDevice:
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return "coreml" if sys.platform == "darwin" else "cpu"
    if requested in {"coreml", "mps", "gpu"} and sys.platform == "darwin":
        return "coreml"
    raise ValueError(
        "Core ML acceleration is available on macOS only; CUDA is not configured. "
        "Use --device cpu or --device auto."
    )


def diarize(audio: Audio, device: InferenceDevice) -> list[SpeakerTurn]:
    import sherpa_onnx

    print(
        f"Identifying speakers with ONNX segmentation and TitaNet using the {device} provider...",
        flush=True,
    )
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=model_file(DIARIZATION_MODEL, "model.onnx"),
            ),
            provider=device,
            num_threads=4,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_file(EMBEDDING_MODEL, EMBEDDING_FILE),
            provider=device,
            num_threads=4,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise ValueError("Invalid ONNX diarization configuration; check the cached model files.")
    model = sherpa_onnx.OfflineSpeakerDiarization(config)
    if model.sample_rate != SAMPLE_RATE:
        raise ValueError("The diarization model must accept 16 kHz audio.")
    return [
        SpeakerTurn(segment.start, segment.end, f"Speaker {segment.speaker + 1}")
        for segment in model.process(audio).sort_by_start_time()
    ]


def _result_words(result: RecognitionResult, offset: float, duration: float) -> list[Word]:
    """Convert SentencePiece tokens and TDT durations into absolute word timestamps."""
    if not result.tokens or len(result.tokens) != len(result.timestamps):
        return [Word(None, None, result.text)] if result.text.strip() else []
    tokens: list[Word] = []
    for index, (token, start) in enumerate(zip(result.tokens, result.timestamps, strict=True)):
        if len(result.durations) == len(result.tokens):
            end = start + result.durations[index]
        else:
            # Some exports supply starts only; use the next start as an estimate.
            end = result.timestamps[index + 1] if index + 1 < len(result.tokens) else duration
        start = max(0.0, min(start, duration))
        end = max(start, min(end, duration))
        tokens.append(Word(offset + start, offset + end, token.replace("▁", " ")))
    return merge_word_tokens(tokens)


def transcribe(
    audio: Audio,
    device: InferenceDevice,
    model_id: str = DEFAULT_ASR_MODEL,
    language: str | None = "en",
) -> list[Word]:
    if language is not None and language.lower() not in {"en", "english"}:
        raise ValueError("Parakeet v2 supports English only.")
    import sherpa_onnx

    print(f"Transcribing English with {model_id} using ONNX provider {device}...", flush=True)
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=model_file(model_id, "encoder.int8.onnx"),
        decoder=model_file(model_id, "decoder.int8.onnx"),
        joiner=model_file(model_id, "joiner.int8.onnx"),
        tokens=model_file(model_id, "tokens.txt"),
        model_type="nemo_transducer",
        sample_rate=SAMPLE_RATE,
        provider=device,
        num_threads=4,
        decoding_method="greedy_search",
    )
    words: list[Word] = []
    for core_start in range(0, len(audio), _CHUNK_SAMPLES):
        core_end = min(core_start + _CHUNK_SAMPLES, len(audio))
        start = max(0, core_start - _CONTEXT_SAMPLES)
        end = min(len(audio), core_end + _CONTEXT_SAMPLES)
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio[start:end])
        recognizer.decode_stream(stream)
        chunk_words = _result_words(stream.result, start / SAMPLE_RATE, (end - start) / SAMPLE_RATE)
        if any(word.start is None for word in chunk_words) and (start, end) != (
            core_start,
            core_end,
        ):
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio[core_start:core_end])
            recognizer.decode_stream(stream)
            chunk_words = _result_words(
                stream.result, core_start / SAMPLE_RATE, (core_end - core_start) / SAMPLE_RATE
            )
        # Each chunk owns words whose midpoint is inside its non-overlapping core.
        for word in chunk_words:
            if word.start is None or word.end is None:
                words.append(word)
            elif core_start / SAMPLE_RATE <= (word.start + word.end) / 2 < core_end / SAMPLE_RATE:
                words.append(word)
    return words
