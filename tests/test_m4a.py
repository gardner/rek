"""Voice Memo decoding, including real AAC and ALAC files when FFmpeg is installed."""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from gkit.audio import SAMPLE_RATE, load_audio


@pytest.mark.parametrize("codec", ["aac", "alac"])
@pytest.mark.parametrize("channels", [1, 2])
def test_loads_m4a_as_16khz_mono(tmp_path: Path, codec: str, channels: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is needed to generate the M4A test fixture")
    source = tmp_path / "source.wav"
    path = tmp_path / "Voice Memo.M4A"
    samples = 0.25 * np.sin(2 * np.pi * 440 * np.arange(44_100) / 44_100)
    sf.write(source, samples if channels == 1 else np.column_stack([samples, samples]), 44_100)
    subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", str(source), "-c:a", codec, str(path)],
        check=True,
        capture_output=True,
    )

    audio = load_audio(path)

    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(audio) == pytest.approx(SAMPLE_RATE, abs=500)
    # Check steady-state energy; lossy AAC can add transients near clip boundaries.
    rms = float(np.sqrt(np.mean(audio[1600:14400] ** 2)))
    assert rms == pytest.approx(0.25 / np.sqrt(2), abs=0.01)
    assert path.is_file()
    assert sorted(file.name for file in tmp_path.iterdir()) == ["Voice Memo.M4A", "source.wav"]


def test_missing_ffmpeg_has_an_actionable_message(tmp_path: Path) -> None:
    path = tmp_path / "memo.m4a"
    path.write_bytes(b"m4a")
    with patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="FFmpeg"):
            load_audio(path)


def test_corrupt_m4a_reports_a_decoding_error(tmp_path: Path) -> None:
    path = tmp_path / "memo.m4a"
    path.write_bytes(b"corrupt audio")
    error = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"Invalid data found")
    with patch("subprocess.run", side_effect=error):
        with pytest.raises(ValueError, match="Could not decode M4A"):
            load_audio(path)


def test_m4a_with_no_audio_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "memo.m4a"
    path.write_bytes(b"m4a")
    result = subprocess.CompletedProcess(["ffmpeg"], 0, stdout=b"", stderr=b"")
    with patch("subprocess.run", return_value=result):
        with pytest.raises(ValueError, match="empty"):
            load_audio(path)
