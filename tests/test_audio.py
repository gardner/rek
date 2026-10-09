"""Audio normalization is shared by microphone and file input."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from tscribe.audio import SAMPLE_RATE, load_audio, record_audio


def test_loads_stereo_at_another_rate_as_16khz_mono(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    sf.write(path, np.full((8000, 2), 0.25), 8000)

    audio = load_audio(path)

    assert audio.shape == (SAMPLE_RATE,)
    assert audio.dtype == np.float32
    assert float(audio[100]) == pytest.approx(0.25, abs=0.01)


def test_rejects_empty_audio(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    sf.write(path, np.empty(0), SAMPLE_RATE)
    with pytest.raises(ValueError, match="empty"):
        load_audio(path)


def test_records_and_saves_audio_before_inference(tmp_path: Path) -> None:
    path = tmp_path / "recording.wav"
    samples = np.full((SAMPLE_RATE, 1), 0.25, dtype=np.float32)
    with patch("sounddevice.rec", return_value=samples) as record:
        audio = record_audio(path, 1.0)

    record.assert_called_once_with(
        SAMPLE_RATE, samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocking=True
    )
    assert audio.shape == (SAMPLE_RATE,)
    saved, rate = sf.read(path)
    assert rate == SAMPLE_RATE
    assert saved[0] == pytest.approx(0.25)


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), 0.000001])
def test_rejects_invalid_durations_before_recording(tmp_path: Path, seconds: float) -> None:
    with patch("sounddevice.rec") as record:
        with pytest.raises(ValueError):
            record_audio(tmp_path / "recording.wav", seconds)
        record.assert_not_called()
