"""Default MLX Parakeet transcription with aligned word timestamps."""

from typing import Literal, Protocol, cast

from rek.audio import Audio
from rek.mlx_diarization import validate_platform
from rek.transcript import Word, merge_word_tokens

DEFAULT_ASR_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"
MODEL_REVISION = "8ae155301e23d820d82aa60d24817c900e69e487"


class Token(Protocol):
    text: str
    start: float
    end: float


class Sentence(Protocol):
    tokens: list[Token]


class Result(Protocol):
    sentences: list[Sentence]


class Model(Protocol):
    def generate(
        self, audio: object, *, chunk_duration: float, overlap_duration: float
    ) -> Result: ...


def transcribe(
    audio: Audio,
    device: Literal["cpu", "coreml"],
    model_id: str = DEFAULT_ASR_MODEL,
    language: str | None = "en",
) -> list[Word]:
    if language is not None and language.lower() not in {"en", "english"}:
        raise ValueError("Parakeet v2 supports English only.")
    validate_platform()
    import mlx.core as mx  # pyright: ignore[reportMissingImports]
    from mlx_audio.stt.utils import load_model  # pyright: ignore[reportMissingImports]

    if not len(audio):
        return []
    target = mx.cpu if device == "cpu" else mx.gpu
    print(f"Transcribing English with {model_id} using MLX...", flush=True)
    with mx.stream(target):
        model = cast(
            Model,
            load_model(
                model_id,
                strict=True,
                revision=MODEL_REVISION if model_id == DEFAULT_ASR_MODEL else None,
            ),
        )
        result = model.generate(mx.array(audio), chunk_duration=30.0, overlap_duration=2.0)
        return merge_word_tokens(
            [
                Word(float(token.start), float(token.end), token.text.replace("▁", " "))
                for sentence in result.sentences
                for token in sentence.tokens
            ]
        )
