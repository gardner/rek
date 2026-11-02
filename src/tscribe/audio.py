"""Capture or normalize audio to the models' common 16 kHz mono input."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

SAMPLE_RATE = 16_000
type Audio = npt.NDArray[np.float32]


def record_audio(path: Path) -> Audio:
    """Write microphone blocks to disk until Ctrl+C, then return the saved audio."""
    import sounddevice as sd

    overflowed = False
    try:
        with (
            sf.SoundFile(
                path, mode="x", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16"
            ) as wav,
            sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream,
        ):
            print("Recording. Speak now! Press Ctrl+C to stop and transcribe.", flush=True)
            try:
                while True:
                    samples, dropped = stream.read(SAMPLE_RATE // 10)
                    overflowed |= dropped
                    wav.write(samples)
            except KeyboardInterrupt:
                pass  # Ctrl+C finishes this recording; model inference continues below.
    except sd.PortAudioError as error:
        raise RuntimeError(
            "Could not record audio. Check your default microphone and microphone permissions."
        ) from error
    if overflowed:
        print("Warning: microphone input overflowed; some audio was lost.", file=sys.stderr)
    if sf.info(path).frames == 0:
        raise ValueError("No audio was recorded. Start again and speak before pressing Ctrl+C.")
    print(f"\nRecording stopped. Audio saved to {path}", flush=True)
    return load_audio(path)


def _load_m4a(path: Path) -> Audio:
    """Decode AAC or ALAC in memory without creating an intermediate WAV."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(path.resolve()),
                "-map",
                "0:a:0",
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-rematrix_maxval",
                "1",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "M4A input requires FFmpeg on PATH. Install FFmpeg with your system package manager "
            "(macOS: brew install ffmpeg; Debian/Ubuntu: sudo apt install ffmpeg; "
            "Windows: winget install Gyan.FFmpeg)."
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip()
        raise ValueError(f"Could not decode M4A audio: {detail}") from error
    if not result.stdout:
        raise ValueError("The audio file is empty.")
    return np.frombuffer(result.stdout, dtype="<f4").astype(np.float32)


def load_audio(path: Path) -> Audio:
    """Load M4A or a soundfile-supported file as 16 kHz mono audio."""
    if path.suffix.lower() == ".m4a":
        return _load_m4a(path)
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError("The audio file is empty.")
    audio = np.asarray(samples.mean(axis=1), dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)
