"""The default installation and pipeline must not depend on ONNX."""

import json
import tomllib
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rek.cli import parse_args, run
from rek.transcript import SpeakerTurn, Word


def test_default_install_excludes_onnx() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert not any("onnx" in dep for dep in project["dependencies"])
    assert any(dep.startswith("mlx-audio") for dep in project["dependencies"])
    assert any(dep.startswith("onnxruntime") for dep in project["optional-dependencies"]["onnx"])
    assert any(dep.startswith("sherpa-onnx") for dep in project["optional-dependencies"]["onnx"])


def test_backend_selects_both_stages_and_matching_model() -> None:
    default = parse_args([])
    assert default.diarization_backend == default.asr_backend == "mlx"
    assert default.asr_model == "mlx-community/parakeet-tdt-0.6b-v2"
    onnx = parse_args(["--backend", "onnx"])
    assert onnx.diarization_backend == onnx.asr_backend == "onnx"
    assert onnx.asr_model.startswith("csukuangfj/")


def test_default_pipeline_uses_no_onnx(tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    audio.touch()
    options = parse_args([str(audio), "--output-dir", str(tmp_path / "out")])
    with (
        patch("rek.cli.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
        patch("rek.mlx_diarization.validate_platform"),
        patch("rek.mlx_diarization.diarize", return_value=[SpeakerTurn(0, 1, "Speaker 1")]),
        patch("rek.mlx_asr.transcribe", return_value=[Word(0, 1, "Hello")]),
        patch("rek.cli.diarize", side_effect=AssertionError("ONNX diarization loaded")),
        patch("rek.cli.transcribe", side_effect=AssertionError("ONNX ASR loaded")),
    ):
        output = run(options)
    assert "Hello" in (output / "transcript.txt").read_text()
    data = json.loads((output / "transcript.json").read_text())
    assert data["backend"] == data["asr_backend"] == "mlx"
