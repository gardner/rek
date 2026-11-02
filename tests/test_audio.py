"""Audio normalization is shared by microphone and file input."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import sounddevice as sd
import soundfile as sf

from rek.audio import SAMPLE_RATE, load_audio, record_audio


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


def test_ctrl_c_stops_recording_and_preserves_all_completed_blocks(tmp_path: Path) -> None:
    path = tmp_path / "recording.wav"
    first = np.full((1600, 1), 0.25, dtype=np.float32)
    second = np.full((1600, 1), -0.25, dtype=np.float32)
    with patch("sounddevice.InputStream") as input_stream:
        stream = input_stream.return_value.__enter__.return_value
        stream.read.side_effect = [(first, False), (second, False), KeyboardInterrupt]
        audio = record_audio(path)

    input_stream.return_value.__exit__.assert_called_once()
    assert audio.shape == (3200,)
    saved, rate = sf.read(path)
    assert rate == SAMPLE_RATE
    np.testing.assert_allclose(saved, np.concatenate([first, second]).reshape(-1))
    np.testing.assert_allclose(audio, saved)


def test_stopping_before_any_audio_reports_an_empty_recording(tmp_path: Path) -> None:
    with patch("sounddevice.InputStream") as input_stream:
        stream = input_stream.return_value.__enter__.return_value
        stream.read.side_effect = KeyboardInterrupt
        with pytest.raises(ValueError, match="No audio was recorded"):
            record_audio(tmp_path / "recording.wav")
    input_stream.return_value.__exit__.assert_called_once()


def test_device_failure_closes_stream_and_preserves_completed_audio(tmp_path: Path) -> None:
    path = tmp_path / "recording.wav"
    samples = np.full((1600, 1), 0.25, dtype=np.float32)
    with patch("sounddevice.InputStream") as input_stream:
        stream = input_stream.return_value.__enter__.return_value
        stream.read.side_effect = [(samples, False), sd.PortAudioError("device disconnected")]
        with pytest.raises(RuntimeError, match="microphone"):
            record_audio(path)
    input_stream.return_value.__exit__.assert_called_once()
    saved, rate = sf.read(path)
    assert rate == SAMPLE_RATE
    np.testing.assert_allclose(saved, samples.reshape(-1))


def test_warns_when_microphone_samples_are_lost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    samples = np.zeros((1600, 1), dtype=np.float32)
    with patch("sounddevice.InputStream") as input_stream:
        stream = input_stream.return_value.__enter__.return_value
        stream.read.side_effect = [(samples, True), KeyboardInterrupt]
        record_audio(tmp_path / "recording.wav")
    assert "audio was lost" in capsys.readouterr().err
