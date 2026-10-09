# tscribe

Record your microphone, then transcribe locally with timestamps and speaker labels.
[NVIDIA Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
detects who spoke when (up to eight speakers).
[Whisper](https://huggingface.co/openai/whisper-base) supplies the words; Nemotron
does not transcribe speech by itself.

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
downloads both models from Hugging Face (hundreds of MB); later runs use the local
cache. Audio is processed on your computer.

```sh
# Record until Ctrl+C, specifying the spoken language.
uv run main.py --language en

# Process an existing file without opening the microphone.
uv run main.py --input meeting.wav

# Transcribe an Apple Voice Memo directly; conversion is automatic.
uv run main.py --input "Voice Memo.m4a"

# Use a larger Whisper model and choose a new output directory.
uv run main.py --asr-model openai/whisper-small --output-dir recordings/meeting

uv run main.py --help
```

M4A (AAC or Apple Lossless), WAV, FLAC, and other formats supported by libsndfile can
be used as input. M4A decoding uses FFmpeg automatically, without changing the
original file or creating an intermediate WAV. FFmpeg must be on PATH; on macOS,
install it if needed with `brew install ffmpeg`. Stereo files are mixed to mono
and resampled to 16 kHz. Language is auto-detected unless `--language` is supplied.
Transcription starts after recording ends.

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
people by name. Words are assigned to the speaker with the greatest total time
overlap. Ties, missing word timestamps, and unmatched words are labelled `Unknown`.
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

`--device auto` uses CUDA when available, otherwise CPU. `--device cpu` works without
an NVIDIA GPU. On Apple Silicon, `--device mps` opts into Metal acceleration; use CPU
if your PyTorch build reports an unsupported MPS operation. CUDA requires a compatible
PyTorch installation and NVIDIA driver.

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
