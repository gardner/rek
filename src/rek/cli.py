"""A small command-line interface around recording and local model inference."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from rek.audio import load_audio, record_audio
from rek.models import (
    DEFAULT_ASR_MODEL,
    DIARIZATION_MODEL,
    Device,
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


def parse_args(argv: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(
        prog="rek",
        description="Transcribe an audio file with timestamps and speaker labels. "
        "Without a file, record your microphone until Ctrl+C.",
        epilog="Example: rek 'Voice Memo.m4a'. Models download automatically on first use. "
        "Defaults: English, Nemotron speaker detection, Parakeet transcription, "
        "Core ML on macOS and CPU elsewhere.",
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
        help="ONNX provider (auto: Core ML on macOS, CPU elsewhere; mps/gpu alias Core ML).",
    )
    advanced.add_argument(
        "--asr-model",
        default=DEFAULT_ASR_MODEL,
        help="Sherpa ONNX Parakeet v2 INT8 repository ID or local model directory.",
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
        args.asr_model,
        args.language,
    )
    if options.input is not None and not options.input.is_file():
        parser.error(f"Audio file does not exist: {options.input}")
    return options


def run(options: Options) -> Path:
    device = resolve_device(options.device)
    output_dir = options.output_dir or Path("recordings") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    audio_path = options.input or output_dir / "recording.wav"
    audio = load_audio(audio_path) if options.input else record_audio(audio_path)
    turns = diarize(audio, device)
    # Skip ASR when diarization detects no speech.
    words = transcribe(audio, device, options.asr_model, options.language) if turns else []
    utterances = align_words(words, turns)
    text = render_transcript(utterances)
    (output_dir / "transcript.txt").write_text(text, encoding="utf-8")
    data = {
        "audio_file": str(audio_path.resolve()),
        "diarization_model": DIARIZATION_MODEL,
        "backend": "onnxruntime",
        "asr_backend": "sherpa-onnx",
        "diarization_file": "onnx/model.onnx",
        "device": device,
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
