"""Capture or normalize audio to the models' common 16 kHz mono input."""

from math import isfinite
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

SAMPLE_RATE = 16_000
type Audio = npt.NDArray[np.float32]


def validate_duration(seconds: float) -> None:
    if not isfinite(seconds) or seconds < 1 / SAMPLE_RATE:
        raise ValueError("Recording duration must be finite and at least one audio sample.")


def record_audio(path: Path, seconds: float) -> Audio:
    """Record a fixed duration, then preserve the WAV before loading any models."""
    validate_duration(seconds)
    import sounddevice as sd

    print(f"Recording for {seconds:g} seconds. Speak now!", flush=True)
    try:
        samples = sd.rec(
            round(seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocking=True,
        )
    except sd.PortAudioError as error:
        raise RuntimeError(
            "Could not record audio. Check your default microphone and microphone permissions."
        ) from error
    finally:
        sd.stop()
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"Audio saved to {path}", flush=True)
    return audio


def load_audio(path: Path) -> Audio:
    """Load a soundfile-supported file, mix channels, and resample if necessary."""
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError("The audio file is empty.")
    audio = np.asarray(samples.mean(axis=1), dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)
