"""A small command-line interface around recording and local model inference."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from tscribe.audio import load_audio, record_audio, validate_duration
from tscribe.models import (
    DEFAULT_ASR_MODEL,
    DIARIZATION_MODEL,
    Device,
    diarize,
    resolve_device,
    transcribe,
)
from tscribe.transcript import align_words, render_transcript


@dataclass(frozen=True)
class Options:
    seconds: float
    input: Path | None
    output_dir: Path | None
    device: Device
    asr_model: str
    language: str | None


def parse_args(argv: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(description="Record and transcribe audio with speaker labels.")
    parser.add_argument(
        "--seconds", type=float, default=30.0, help="Recording duration (default: 30)."
    )
    parser.add_argument("--input", type=Path, help="Transcribe an existing audio file instead.")
    parser.add_argument(
        "--output-dir", type=Path, help="New output directory (default: recordings/<time>)."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Inference device (auto: CUDA if available, otherwise CPU).",
    )
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL, help="Whisper model ID.")
    parser.add_argument("--language", help="Spoken language, e.g. en; auto-detected when omitted.")
    args = parser.parse_args(argv)
    options = Options(
        args.seconds,
        args.input,
        args.output_dir,
        cast(Device, args.device),
        args.asr_model,
        args.language,
    )
    try:
        validate_duration(options.seconds)
    except ValueError as error:
        parser.error(str(error))
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
    audio = load_audio(audio_path) if options.input else record_audio(audio_path, options.seconds)
    turns = diarize(audio, device)
    # Avoid Whisper hallucinations on recordings where Nemotron detects no speech.
    words = transcribe(audio, device, options.asr_model, options.language) if turns else []
    utterances = align_words(words, turns)
    text = render_transcript(utterances)
    (output_dir / "transcript.txt").write_text(text, encoding="utf-8")
    data = {
        "audio_file": str(audio_path.resolve()),
        "diarization_model": DIARIZATION_MODEL,
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
