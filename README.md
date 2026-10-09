# tscribe

Record your microphone, then transcribe locally with timestamps and speaker labels.
[Nemotron-3-Diarization-8bit in MLX](https://huggingface.co/mlx-community/Nemotron-3-Diarization-8bit)
detects who spoke when (up to eight speakers), using MLX Audio on the Apple GPU.
[Parakeet TDT 0.6B v2 in MLX](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2)
transcribes **English only**. Nemotron does not transcribe speech by itself.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv run main.py
```

This installs the dependencies and records from your default microphone until you
press **Ctrl+C**. Start speaking when you see **“Recording”**. Ctrl+C stops recording,
finishes saving the WAV, and starts transcription. There is no duration limit or
`--seconds` option.
The project uses Python 3.14; uv can install it automatically. The first inference
downloads both models from Hugging Face (roughly 2.6 GB total); later runs use the
local cache. Audio is processed on your computer.

```sh
# Record English until Ctrl+C, explicitly using the Apple GPU.
uv run main.py --device gpu

# Process an existing file without opening the microphone.
uv run main.py --input meeting.wav

# Transcribe an Apple Voice Memo directly; conversion is automatic.
uv run main.py --input "Voice Memo.m4a"

# Choose a new output directory.
uv run main.py --input meeting.wav --output-dir recordings/meeting

uv run main.py --help
```

M4A (AAC or Apple Lossless), WAV, FLAC, and other formats supported by libsndfile can
be used as input. M4A decoding uses FFmpeg automatically, without changing the
original file or creating an intermediate WAV. FFmpeg must be on PATH; on macOS,
install it if needed with `brew install ffmpeg`. Stereo files are mixed to mono
and resampled to 16 kHz. Parakeet v2 is English-only; `--language en` is optional,
and other languages are rejected. Transcription starts after recording ends.
Long audio is decoded in overlapping 30-second chunks, then merged with timestamps
relative to the full recording.

## Output

Each run creates a new `recordings/<timestamp>/` directory containing:

- `recording.wav`: 16 kHz mono audio (microphone mode only).
- `transcript.txt`: readable timestamped speaker turns, also printed to the terminal.
- `transcript.json`: transcript, individual word timestamps, raw speaker turns,
  model IDs, and the original audio path.

Example format:

```text
[00:00:00.100 - 00:00:02.300] Speaker 1: Hello, everyone.
[00:00:02.500 - 00:00:03.800] Speaker 2: Hi there.
```

Speaker labels are anonymous and apply to one recording; they do not identify
people by name. Parakeet's timestamped subword pieces are joined into complete
words, preserving punctuation. Words are assigned to the speaker with the greatest
total time overlap. Ties, missing word timestamps, and unmatched words are labelled `Unknown`.
Overlapping voices can reduce accuracy: this script does not separate mixed voices.
When Nemotron detects no speech, transcription is skipped. Microphone audio is
written to disk as you record; the completed recording is loaded into memory for
inference. CPU inference can take longer than the recording.

Existing output directories are refused to prevent overwriting recordings. If model
loading or inference fails, a completed microphone recording remains available;
retry it with `--input path/to/recording.wav`. Ctrl+C during recording saves and
transcribes the captured audio; Ctrl+C during model loading or transcription cancels
processing and preserves the recording. Stopping before any audio arrives reports
an empty recording instead of starting inference.

## Devices and troubleshooting

On Apple Silicon, `--device auto` runs both Parakeet and Nemotron through MLX's
Metal GPU backend. `--device gpu` explicitly selects the GPU, and `--device mps`
remains an alias for the Metal GPU. `--device cpu` forces both models onto CPU.
Auto selection uses CPU if MLX cannot access a GPU. PyTorch is no longer required.
This setup is tested on Apple Silicon macOS.

If microphone capture fails, check the system's default input device and grant
microphone access to your terminal/app. macOS and Windows sounddevice wheels include
PortAudio. Linux may need `libportaudio2` and `libsndfile1` installed using the system
package manager. If a device does not support 16 kHz mono capture, record a file
with another recorder and pass it with `--input`.

## Development

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run python scripts/check_file_length.py
```

Tests use synthetic audio and mocked recording/inference; they do not access a
microphone or download model weights. M4A integration tests generate real AAC and
ALAC files and are skipped when FFmpeg is unavailable. Ruff enforces annotations
and a complexity limit of 10, Pyright checks types, and source files are limited to
500 lines.
