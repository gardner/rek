"""Check CLI orchestration without downloading models or opening a microphone."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tscribe.cli import Options, main, parse_args, run
from tscribe.transcript import SpeakerTurn, Word


def test_defaults_to_english_parakeet_v2() -> None:
    options = parse_args([])
    assert options.asr_model == "mlx-community/parakeet-tdt-0.6b-v2"
    assert options.language == "en"


def test_non_english_language_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as error:
        parse_args(["--language", "fr"])
    assert error.value.code == 2


def test_seconds_option_has_been_removed() -> None:
    with pytest.raises(SystemExit) as error:
        parse_args(["--seconds", "30"])
    assert error.value.code == 2


def test_missing_input_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        parse_args(["--input", str(tmp_path / "absent.wav")])
    assert error.value.code == 2


def test_writes_transcript_and_diarization_metadata(tmp_path: Path) -> None:
    output = tmp_path / "result"
    options = Options(None, output, "cpu", "mlx-community/parakeet-tdt-0.6b-v2", "en")
    samples = np.zeros(16_000, dtype=np.float32)
    with (
        patch("tscribe.cli.resolve_device", return_value="cpu"),
        patch("tscribe.cli.record_audio", return_value=samples) as record,
        patch("tscribe.cli.diarize", return_value=[SpeakerTurn(0.0, 1.0, "Speaker 1")]),
        patch("tscribe.cli.transcribe", return_value=[Word(0.1, 0.8, " Hello.")]),
    ):
        assert run(options) == output
    record.assert_called_once_with(output / "recording.wav")
    assert "Speaker 1: Hello." in (output / "transcript.txt").read_text()
    data = json.loads((output / "transcript.json").read_text())
    assert data["diarization_model"] == "mlx-community/Nemotron-3-Diarization-8bit"
    assert data["asr_model"] == "mlx-community/parakeet-tdt-0.6b-v2"
    assert data["transcript"][0]["text"] == "Hello."
    assert data["speaker_turns"][0]["speaker"] == "Speaker 1"


def test_file_input_skips_recording_and_silence_skips_asr(tmp_path: Path) -> None:
    options = Options(tmp_path / "input.wav", tmp_path / "out", "cpu", "unused", None)
    with (
        patch("tscribe.cli.resolve_device", return_value="cpu"),
        patch("tscribe.cli.load_audio", return_value=np.zeros(16_000, dtype=np.float32)),
        patch("tscribe.cli.record_audio") as record,
        patch("tscribe.cli.diarize", return_value=[]),
        patch("tscribe.cli.transcribe") as asr,
    ):
        output = run(options)
    record.assert_not_called()
    asr.assert_not_called()
    assert (output / "transcript.txt").read_text() == "No speech detected.\n"


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    options = Options(None, tmp_path, "cpu", "unused", None)
    with patch("tscribe.cli.resolve_device", return_value="cpu"):
        with pytest.raises(FileExistsError):
            run(options)


def test_recording_stop_continues_through_transcription(tmp_path: Path) -> None:
    options = Options(None, tmp_path / "out", "cpu", "mlx-community/parakeet-tdt-0.6b-v2", "en")
    with (
        patch("tscribe.cli.resolve_device", return_value="cpu"),
        patch("sounddevice.InputStream") as input_stream,
        patch("tscribe.cli.diarize", return_value=[SpeakerTurn(0.0, 0.1, "Speaker 1")]),
        patch("tscribe.cli.transcribe", return_value=[Word(0.0, 0.1, " Hello.")]) as asr,
    ):
        stream = input_stream.return_value.__enter__.return_value
        stream.read.side_effect = [
            (np.zeros((1600, 1), dtype=np.float32), False),
            KeyboardInterrupt,
        ]
        output = run(options)
    asr.assert_called_once()
    assert (output / "recording.wav").is_file()
    assert "Speaker 1: Hello." in (output / "transcript.txt").read_text()


def test_ctrl_c_during_processing_still_cancels() -> None:
    with patch("tscribe.cli.run", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit) as error:
            main([])
    assert error.value.code == 130
