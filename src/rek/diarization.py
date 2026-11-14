"""Nemotron 3 ONNX diarization with cached speaker identity across chunks."""

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from onnxruntime import InferenceSession  # pyright: ignore[reportMissingImports]

import numpy as np

from rek.audio import SAMPLE_RATE, Audio
from rek.model_files import DIARIZATION_MODEL, model_file
from rek.speaker_cache import SUBSAMPLING, FloatArray, SpeakerCache
from rek.transcript import SpeakerTurn

_HOP = 160
_CHUNK_FRAMES = 340 * SUBSAMPLING
_LOOKAHEAD_FRAMES = 40 * SUBSAMPLING


def mel_features(audio: Audio) -> FloatArray:
    """Match the export's unnormalized 128-bin log-mel frontend without Torch.

    See transformers' NemotronAsrStreamingFeatureExtractor: symmetric Hann window,
    constant centering, preemphasis, Slaney filters, and a 2**-24 log guard.
    """
    import librosa

    emphasized = audio.copy()
    emphasized[1:] -= np.float32(0.97) * audio[:-1]
    spectrum = librosa.stft(
        emphasized,
        n_fft=512,
        hop_length=_HOP,
        win_length=400,
        window=np.hanning(400).astype(np.float32),
        center=True,
        pad_mode="constant",
    )
    filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=512, n_mels=128, norm="slaney")
    power = np.abs(spectrum) ** 2
    features = np.log(filters @ power + 2**-24).T
    # The upstream mask counts floor(samples / hop) valid frames.
    return np.asarray(features[: len(audio) // _HOP], dtype=np.float32)


def speaker_turns(logits: FloatArray, duration: float) -> list[SpeakerTurn]:
    """Threshold logits at zero (sigmoid > 0.5), retaining overlapping speakers."""
    changes = np.diff(np.pad((logits > 0).astype(np.int8), ((1, 1), (0, 0))), axis=0)
    turns: list[SpeakerTurn] = []
    for speaker in range(logits.shape[1]):
        starts = np.flatnonzero(changes[:, speaker] == 1)
        ends = np.flatnonzero(changes[:, speaker] == -1)
        for start, end in zip(starts, ends, strict=True):
            turns.append(
                SpeakerTurn(
                    float(start) * 0.01, min(float(end) * 0.01, duration), f"Speaker {speaker + 1}"
                )
            )
    return sorted(turns, key=lambda turn: (turn.start, turn.speaker))


def load_diarizer(device: Literal["cpu", "coreml"]) -> Callable[[Audio], list[SpeakerTurn]]:
    try:
        import onnxruntime as ort  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError(
            "ONNX dependencies are missing. Install with: uv tool install 'rek[onnx]' "
            "(from a checkout: uv sync --extra onnx)."
        ) from error

    print(f"Identifying speakers with {DIARIZATION_MODEL} using {device}...", flush=True)
    # The graph references this sibling external-data file by its exact basename.
    graph = model_file(DIARIZATION_MODEL, "onnx/model.onnx")
    model_file(DIARIZATION_MODEL, "onnx/model.onnx_data")
    providers = ["CPUExecutionProvider"]
    if device == "coreml":
        if "CoreMLExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CoreMLExecutionProvider")
        else:
            print("Core ML is unavailable in ONNX Runtime; diarization uses CPU.", file=sys.stderr)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(graph, sess_options=options, providers=providers)
    return lambda audio: _infer(audio, session)


def _infer(audio: Audio, session: InferenceSession) -> list[SpeakerTurn]:
    if len(audio) < _HOP:
        return []
    features = mel_features(audio)
    cache = SpeakerCache()
    outputs: list[FloatArray] = []
    for start in range(0, len(features), _CHUNK_FRAMES):
        end = min(start + _CHUNK_FRAMES, len(features))
        chunk = features[start : end + _LOOKAHEAD_FRAMES]
        cached = cache.embeds
        frame_count = (len(chunk) + SUBSAMPLING - 1) // SUBSAMPLING
        logits, embeds, silence = cast(
            list[FloatArray],
            session.run(
                ["logits", "chunk_embeds", "silence_embeds"],
                {
                    "input_features": chunk[None],
                    "cached_embeds": cached[None],
                    "attention_mask": np.ones((1, len(cached) + frame_count), dtype=np.int64),
                },
            ),
        )
        core_frames = (end - start + SUBSAMPLING - 1) // SUBSAMPLING
        offset = len(cached) * SUBSAMPLING
        outputs.append(logits[0, offset : offset + end - start])
        cache.update(embeds[0, :core_frames], logits[0], silence)
    return speaker_turns(np.concatenate(outputs), len(audio) / SAMPLE_RATE)


def diarize(audio: Audio, device: Literal["cpu", "coreml"]) -> list[SpeakerTurn]:
    if len(audio) < _HOP:
        return []
    return load_diarizer(device)(audio)
