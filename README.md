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

This installs the dependencies, records **30 seconds** from your default microphone,
and saves the audio before processing it. Start speaking when you see **“Recording”**.
The project uses Python 3.14; uv can install it automatically. The first inference
downloads both models from Hugging Face (hundreds of MB); later runs use the local
cache. Audio is processed on your computer.

```sh
# Record for a minute, specifying the spoken language.
uv run main.py --seconds 60 --language en

# Process an existing file without opening the microphone.
uv run main.py --input meeting.wav

# Use a larger Whisper model and choose a new output directory.
uv run main.py --asr-model openai/whisper-small --output-dir recordings/meeting

uv run main.py --help
```

WAV, FLAC, and other formats supported by libsndfile can be used as input. Stereo
files are mixed to mono and resampled to 16 kHz. Language is auto-detected unless
`--language` is supplied. Transcription starts after recording ends.

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
When Nemotron detects no speech, transcription is skipped. Recordings are held in
memory, so start with short clips; CPU inference can take longer than the recording.

Existing output directories are refused to prevent overwriting recordings. If model
loading or inference fails, a completed microphone recording remains available;
retry it with `--input path/to/recording.wav`. Ctrl+C cancels the current operation;
cancelling during recording discards the incomplete recording.

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
microphone or download model weights. Ruff enforces annotations and a complexity
limit of 10, Pyright checks types, and source files are limited to 500 lines.
