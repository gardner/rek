"""Platform defaults select matching runtimes, models, and dependencies."""

import json
import tomllib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from packaging.requirements import Requirement

from rek.cli import parse_args, run
from rek.transcript import SpeakerTurn, Word

PLATFORMS = [
    ("darwin", "arm64", "mlx"),
    ("darwin", "x86_64", "onnx"),
    ("linux", "x86_64", "onnx"),
    ("linux", "aarch64", "onnx"),
    ("win32", "AMD64", "onnx"),
    ("win32", "ARM64", "onnx"),
]


@pytest.mark.parametrize("system,machine,expected", PLATFORMS)
def test_default_install_matches_platform(system: str, machine: str, expected: str) -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    requirements = [Requirement(dep) for dep in project["dependencies"]]
    installed = {
        dep.name
        for dep in requirements
        if dep.marker is None
        or dep.marker.evaluate({"sys_platform": system, "platform_machine": machine})
    }
    assert ("mlx-audio" in installed) == (expected == "mlx")
    assert {"onnxruntime", "sherpa-onnx"} <= installed
    assert "onnx" not in project.get("optional-dependencies", {})


@pytest.mark.parametrize("system,machine,expected", PLATFORMS)
@pytest.mark.parametrize("flags", [[], ["--backend", "auto"]])
def test_platform_selects_both_stages_and_model(
    system: str, machine: str, expected: str, flags: list[str]
) -> None:
    with patch("sys.platform", system), patch("platform.machine", return_value=machine):
        options = parse_args(flags)
    assert options.diarization_backend == options.asr_backend == expected
    prefix = "mlx-community/" if expected == "mlx" else "csukuangfj/"
    assert options.asr_model.startswith(prefix)


@pytest.mark.parametrize("system,machine,_expected", PLATFORMS)
@pytest.mark.parametrize("backend", ["mlx", "onnx"])
def test_explicit_backend_overrides_platform(
    system: str, machine: str, _expected: str, backend: str
) -> None:
    with patch("sys.platform", system), patch("platform.machine", return_value=machine):
        options = parse_args(["--backend", backend])
    assert options.diarization_backend == options.asr_backend == backend


def test_stage_overrides_and_custom_model_take_precedence() -> None:
    options = parse_args(
        [
            "--backend",
            "onnx",
            "--asr-backend",
            "mlx",
            "--diarization-backend",
            "onnx",
            "--asr-model",
            "custom/model",
        ]
    )
    assert options.asr_backend == "mlx"
    assert options.diarization_backend == "onnx"
    assert options.asr_model == "custom/model"


def test_default_pipeline_uses_no_onnx(tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    audio.touch()
    with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
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


@pytest.mark.parametrize("system,machine,_expected", PLATFORMS[1:])
def test_non_apple_default_runs_onnx(
    tmp_path: Path, system: str, machine: str, _expected: str
) -> None:
    audio = tmp_path / "input.wav"
    audio.touch()
    with patch("sys.platform", system), patch("platform.machine", return_value=machine):
        options = parse_args([str(audio), "--output-dir", str(tmp_path / "out")])
        with (
            patch("rek.cli.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("rek.cli.diarize", return_value=[SpeakerTurn(0, 1, "Speaker 1")]),
            patch("rek.cli.transcribe", return_value=[Word(0, 1, "Hello")]),
            patch("rek.mlx_diarization.diarize", side_effect=AssertionError("MLX loaded")),
            patch("rek.mlx_asr.transcribe", side_effect=AssertionError("MLX loaded")),
        ):
            output = run(options)
    data = json.loads((output / "transcript.json").read_text())
    assert data["backend"] == "onnxruntime"
    assert data["asr_backend"] == "sherpa-onnx"
    assert data["device"] == ("coreml" if system == "darwin" else "cpu")


@pytest.mark.parametrize("module", ["onnxruntime", "sherpa_onnx"])
def test_missing_onnx_explains_repair(module: str) -> None:
    from rek.models import diarize, transcribe

    infer = diarize if module == "onnxruntime" else transcribe
    with patch.dict("sys.modules", {module: None}):
        with pytest.raises(RuntimeError, match=r"uv tool install --reinstall rek"):
            infer(np.zeros(16000, dtype=np.float32), "cpu")
