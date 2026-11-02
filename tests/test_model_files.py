"""Model asset resolution without network access."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rek.model_files import DEFAULT_ASR_MODEL, model_file


def test_local_directory_uses_existing_file(tmp_path: Path) -> None:
    asset = tmp_path / "encoder.int8.onnx"
    asset.write_bytes(b"placeholder")
    with patch("huggingface_hub.hf_hub_download") as download:
        assert model_file(str(tmp_path), asset.name) == str(asset.resolve())
    download.assert_not_called()


def test_local_directory_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing encoder.int8.onnx"):
        model_file(str(tmp_path), "encoder.int8.onnx")


def test_default_model_download_pins_revision() -> None:
    with patch("huggingface_hub.hf_hub_download", return_value="cached.onnx") as download:
        assert model_file(DEFAULT_ASR_MODEL, "encoder.int8.onnx") == "cached.onnx"
    download.assert_called_once_with(
        DEFAULT_ASR_MODEL,
        "encoder.int8.onnx",
        revision="1ab9323565ddb038682214b292f588070a538ce2",
    )


def test_download_failure_has_model_context() -> None:
    with patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")):
        with pytest.raises(RuntimeError, match="Could not load encoder.int8.onnx"):
            model_file(DEFAULT_ASR_MODEL, "encoder.int8.onnx")
