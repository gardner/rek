"""A small command-line interface around recording and local model inference."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from rek.audio import load_audio, record_audio
from rek.mlx_asr import DEFAULT_ASR_MODEL as MLX_ASR_MODEL
from rek.mlx_diarization import MODEL_ID as MLX_DIARIZATION_MODEL
from rek.models import (
    DEFAULT_ASR_MODEL,
    DIARIZATION_MODEL,
    Device,
    InferenceDevice,
    diarize,
    resolve_device,
    transcribe,
)
from rek.transcript import align_words, render_transcript


@dataclass(frozen=True)
class Options:
    input: Path | None
    output_dir: Path | None
    device: Device
    asr_model: str
    language: str | None
    diarization_backend: Literal["onnx", "mlx"] = "mlx"
    asr_backend: Literal["onnx", "mlx"] = "mlx"


def parse_args(argv: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(
        prog="rek",
        description="Transcribe an audio file with timestamps and speaker labels. "
        "Without a file, record your microphone until Ctrl+C.",
        epilog="Example: rek 'Voice Memo.m4a'. Models download automatically on first use. "
        "Defaults: English, Nemotron speaker detection, Parakeet transcription, "
        "MLX on Apple Silicon. Install rek[onnx] for optional ONNX support.",
    )
    parser.add_argument("file", nargs="?", type=Path, help="Audio file, including M4A.")
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-dir", type=Path, help="New output directory (default: recordings/<time>)."
    )
    advanced = parser.add_argument_group("optional overrides (defaults need no flags)")
    advanced.add_argument(
        "--device",
        choices=["auto", "gpu", "cpu", "cuda", "mps", "coreml"],
        default="auto",
        help="Device (auto: Core ML for ONNX on macOS, GPU for MLX; cpu forces CPU).",
    )
    advanced.add_argument(
        "--backend",
        choices=["mlx", "onnx"],
        default="mlx",
        help="Backend for transcription and speaker detection (default: mlx).",
    )
    advanced.add_argument(
        "--asr-backend",
        choices=["mlx", "onnx"],
        help="Override the transcription backend only.",
    )
    advanced.add_argument(
        "--diarization-backend",
        choices=["onnx", "mlx"],
        help="Override the speaker detection backend only.",
    )
    advanced.add_argument(
        "--asr-model",
        help="Compatible Parakeet model repository or directory (default matches backend).",
    )
    advanced.add_argument(
        "--language",
        type=str.lower,
        choices=["en", "english"],
        default="en",
        help="Parakeet v2 is English-only (default: en).",
    )
    args = parser.parse_args(argv)
    if args.file is not None and args.input is not None:
        parser.error("Specify the audio file once: either a filename or --input, not both.")
    options = Options(
        args.file if args.file is not None else args.input,
        args.output_dir,
        cast(Device, args.device),
        args.asr_model
        or (MLX_ASR_MODEL if (args.asr_backend or args.backend) == "mlx" else DEFAULT_ASR_MODEL),
        args.language,
        args.diarization_backend or args.backend,
        args.asr_backend or args.backend,
    )
    if options.input is not None and not options.input.is_file():
        parser.error(f"Audio file does not exist: {options.input}")
    return options


def backend_device(device: InferenceDevice, backend: str) -> str:
    """Report MLX GPU selection without labelling it as the ONNX Core ML provider."""
    return "gpu" if device == "coreml" and backend == "mlx" else device


def run(options: Options) -> Path:
    device = resolve_device(options.device)
    if "mlx" in (options.diarization_backend, options.asr_backend):
        from rek.mlx_diarization import validate_platform

        validate_platform()
    output_dir = options.output_dir or Path("recordings") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    audio_path = options.input or output_dir / "recording.wav"
    audio = load_audio(audio_path) if options.input else record_audio(audio_path)
    if options.diarization_backend == "mlx":
        from rek.mlx_diarization import diarize as diarize_mlx

        turns = diarize_mlx(audio, device)
    else:
        turns = diarize(audio, device)
    # Skip ASR when diarization detects no speech.
    transcribe_audio = transcribe
    if options.asr_backend == "mlx":
        from rek.mlx_asr import transcribe as transcribe_audio
    words = transcribe_audio(audio, device, options.asr_model, options.language) if turns else []
    utterances = align_words(words, turns)
    text = render_transcript(utterances)
    (output_dir / "transcript.txt").write_text(text, encoding="utf-8")
    data = {
        "audio_file": str(audio_path.resolve()),
        "diarization_model": (
            MLX_DIARIZATION_MODEL if options.diarization_backend == "mlx" else DIARIZATION_MODEL
        ),
        "backend": "mlx" if options.diarization_backend == "mlx" else "onnxruntime",
        "diarization_device": backend_device(device, options.diarization_backend),
        "asr_backend": "mlx" if options.asr_backend == "mlx" else "sherpa-onnx",
        "asr_device": backend_device(device, options.asr_backend),
        "diarization_file": (
            "model.safetensors" if options.diarization_backend == "mlx" else "onnx/model.onnx"
        ),
        "device": backend_device(device, options.asr_backend),
        "asr_model": options.asr_model,
        "speaker_turns": [asdict(turn) for turn in turns],
        "words": [asdict(word) for word in words],
        "transcript": [asdict(utterance) for utterance in utterances],
    }
    (output_dir / "transcript.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n{text}\nResults saved to {output_dir.resolve()}")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    try:
        run(options)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
