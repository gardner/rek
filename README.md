# tscribe

Record your microphone, then transcribe locally with timestamps and speaker labels.
Inference uses ONNX Runtime (with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for ASR),
with Core ML acceleration by default on macOS and CPU inference on Linux/Windows.
There is no MLX, PyTorch, or CUDA dependency.

- [Parakeet TDT 0.6B v2 INT8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8)
  transcribes **English only**.
- [Nemotron 3 Diarization ONNX](https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX)
  detects up to eight speakers directly, replacing both Pyannote segmentation and
  TitaNet embeddings/clustering. It runs in 27.2-second chunks with 3.2 seconds of
  look-ahead and an arrival-order speaker cache carried across the full recording.
  The NumPy/librosa frontend and cache follow the model's Transformers implementation;
  PyTorch is not needed. Speaker activity is thresholded at 0.5 on 10 ms frames,
  including overlapping speakers.

Nemotron uses the FP32 graph and its external weights (about 398 MB), pinned to
revision `353b6f8ad2cac3580e982d7fbdf0a010786b0406`. Both files are downloaded into
the same cache directory so ONNX Runtime can resolve the external weights.
The model is distributed under the [OpenMDW 1.1 license](https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX/blob/main/README.md).


The dependencies are intended for macOS, Linux, and Windows. This ONNX migration
has **not been runtime-tested**, including on Linux or Windows.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv run main.py
```

This installs dependencies and records from your default microphone until you
press **Ctrl+C**. Start speaking when you see **“Recording”**. Ctrl+C stops recording,
finishes saving the WAV, and starts transcription. There is no duration limit or
`--seconds` option.

The project uses Python 3.14; uv can install it automatically. The first inference
downloads the required ONNX files from Hugging Face; later runs use its local cache.
Built-in model revisions are pinned. Audio stays on your computer. To relocate the
model cache, set `HF_HOME` before running. For offline use, populate the cache first,
then set `HF_HUB_OFFLINE=1`.

```sh
# Process an existing file without opening the microphone.
uv run main.py --input meeting.wav

# Transcribe an Apple Voice Memo directly; conversion is automatic.
uv run main.py --input "Voice Memo.m4a"

# Choose a new output directory.
uv run main.py --input meeting.wav --output-dir recordings/meeting

# Explicit CPU selection on any platform.
uv run main.py --input meeting.wav --device cpu

# Explicit Core ML acceleration on macOS (mps and gpu are aliases).
uv run main.py --input meeting.wav --device coreml

uv run main.py --help
```

`--asr-model` accepts a compatible sherpa-onnx Parakeet v2 INT8 Hugging Face
repository or local directory containing `encoder.int8.onnx`, `decoder.int8.onnx`,
`joiner.int8.onnx`, and `tokens.txt`. MLX repositories are no longer compatible.

M4A (AAC or Apple Lossless), WAV, FLAC, and other formats supported by libsndfile can
be used as input. M4A decoding uses FFmpeg automatically, without changing the
original file or creating an intermediate WAV. FFmpeg must be on PATH:

- macOS: `brew install ffmpeg`
- Debian/Ubuntu: `sudo apt install ffmpeg`
- Windows: `winget install Gyan.FFmpeg` (reopen your terminal after installation)

Stereo files are mixed to mono and resampled to 16 kHz. Parakeet v2 is English-only;
`--language en` is optional, and other languages are rejected. Transcription starts
after recording ends. Long audio is decoded in 30-second windows with two seconds
of context on either side. Words are retained by their timestamp midpoint within
each window, with timestamps relative to the full recording. Chunk boundaries
can still affect recognition.

## Output

Each run creates a new `recordings/<timestamp>/` directory containing:

- `recording.wav`: 16 kHz mono audio (microphone mode only).
- `transcript.txt`: readable timestamped speaker turns, also printed to the terminal.
- `transcript.json`: transcript, individual word timestamps, raw speaker turns,
  model IDs, diarization graph filename, backends, requested device, and the original audio path.

Example format:

```text
[00:00:00.100 - 00:00:02.300] Speaker 1: Hello, everyone.
[00:00:02.500 - 00:00:03.800] Speaker 2: Hi there.
```

Speaker labels are anonymous and apply to one recording; they do not identify
people by name. Parakeet's timestamped subword pieces are joined into complete
words, preserving punctuation. Words are assigned to the speaker with the greatest
total time overlap. Ties, missing word timestamps, and unmatched words are labelled
`Unknown`. Token end times use TDT durations when available; otherwise they are
estimated from the next token's start or the end of the chunk.
Overlapping voices can reduce accuracy: this script does not separate mixed voices.
When diarization detects no speech, transcription is skipped. Microphone audio is
written to disk as you record; the completed recording is loaded into memory for
inference. CPU inference can take longer than the recording.

Existing output directories are refused to prevent overwriting recordings. If model
loading or inference fails, a completed microphone recording remains available;
retry it with `--input path/to/recording.wav`. Ctrl+C during recording saves and
transcribes the captured audio; Ctrl+C during model loading or transcription cancels
processing and preserves the recording. Stopping before any audio arrives reports
an empty recording instead of starting inference.

## Devices and troubleshooting

`--device auto` requests the ONNX Runtime Core ML provider on macOS for both
transcription and diarization, and CPU elsewhere. `--device coreml`, `--device mps`,
and `--device gpu` explicitly request Core ML on macOS. `--device cpu` forces CPU
on every platform. CUDA is not configured in this project.

[Core ML](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
can use Apple hardware acceleration; it is not PyTorch's MPS backend. Core ML chooses
the compute hardware, and unsupported ONNX operations run on CPU. In particular,
quantized Parakeet operations may remain on CPU: selecting Core ML does not guarantee
that the whole model runs on the GPU or that inference is faster. The bundled macOS
runtime includes Core ML support. Nemotron uses the Python ONNX Runtime package
and falls back to CPU with a warning if Core ML is unavailable in that runtime.
Custom sherpa-onnx runtimes without Core ML may also fall back to CPU.
The JSON `device` field records the requested provider, not measured hardware usage.
Core ML model compilation can add startup time. Use `--device cpu` if initialization
fails or acceleration performs poorly. No acceleration benchmarks were run.

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
ALAC files and are skipped when FFmpeg is unavailable. Tests were updated for ONNX
but deliberately not run during the migration. Ruff enforces annotations and a
complexity limit of 10, Pyright checks types, and source files are limited to 500 lines.
