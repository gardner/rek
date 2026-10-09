"""Check CLI orchestration without downloading models or opening a microphone."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tscribe.cli import Options, parse_args, run
from tscribe.transcript import SpeakerTurn, Word


@pytest.mark.parametrize("duration", ["0", "-2", "nan", "inf"])
def test_invalid_duration_is_a_usage_error(duration: str) -> None:
    with pytest.raises(SystemExit) as error:
        parse_args(["--seconds", duration])
    assert error.value.code == 2


def test_missing_input_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        parse_args(["--input", str(tmp_path / "absent.wav")])
    assert error.value.code == 2


def test_writes_transcript_and_diarization_metadata(tmp_path: Path) -> None:
    output = tmp_path / "result"
    options = Options(1.0, None, output, "cpu", "openai/whisper-base", "en")
    samples = np.zeros(16_000, dtype=np.float32)
    with (
        patch("tscribe.cli.resolve_device", return_value="cpu"),
        patch("tscribe.cli.record_audio", return_value=samples) as record,
        patch("tscribe.cli.diarize", return_value=[SpeakerTurn(0.0, 1.0, "Speaker 1")]),
        patch("tscribe.cli.transcribe", return_value=[Word(0.1, 0.8, " Hello.")]),
    ):
        assert run(options) == output
    record.assert_called_once_with(output / "recording.wav", 1.0)
    assert "Speaker 1: Hello." in (output / "transcript.txt").read_text()
    data = json.loads((output / "transcript.json").read_text())
    assert data["diarization_model"] == "nvidia/Nemotron-3-Diarization"
    assert data["transcript"][0]["text"] == "Hello."
    assert data["speaker_turns"][0]["speaker"] == "Speaker 1"


def test_file_input_skips_recording_and_silence_skips_asr(tmp_path: Path) -> None:
    options = Options(1.0, tmp_path / "input.wav", tmp_path / "out", "cpu", "unused", None)
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
    options = Options(1.0, None, tmp_path, "cpu", "unused", None)
    with patch("tscribe.cli.resolve_device", return_value="cpu"):
        with pytest.raises(FileExistsError):
            run(options)
