"""Download only required ONNX assets, or use a supplied local model directory."""

from pathlib import Path

DEFAULT_ASR_MODEL = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
DIARIZATION_MODEL = "onnx-community/Nemotron-3-Diarization-ONNX"

# Keep the built-in exports reproducible; custom repositories use their default revision.
_REVISIONS = {
    DEFAULT_ASR_MODEL: "1ab9323565ddb038682214b292f588070a538ce2",
    DIARIZATION_MODEL: "353b6f8ad2cac3580e982d7fbdf0a010786b0406",
}


def model_file(model_id: str, filename: str) -> str:
    """Resolve an asset without downloading unrelated weights or sample recordings."""
    directory = Path(model_id).expanduser()
    if directory.is_dir():
        path = directory / filename
        if not path.is_file():
            raise ValueError(f"Model directory is missing {filename}: {directory}")
        return str(path.resolve())

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(model_id, filename, revision=_REVISIONS.get(model_id))
    except Exception as error:
        raise RuntimeError(f"Could not load {filename} from {model_id}: {error}") from error
