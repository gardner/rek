# rek

[GitHub repository](https://github.com/gardner/rek) · [Issues](https://github.com/gardner/rek/issues)

Record your microphone, then transcribe locally with timestamps and speaker labels.
Inference uses ONNX Runtime (with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for ASR),
with Core ML acceleration by default on macOS and CPU inference on Linux/Windows.
MLX diarization is available as an optional extra on Apple Silicon; the default
installation does not require MLX, PyTorch, or CUDA.

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
revision `353b6f8ad2cac3580e982d7fbdf0a010786b0406`. Both files are downloaded as regular files into a revision-specific directory under
the Hugging Face cache. This keeps external weights beside the graph and avoids
ONNX Runtime rejecting the shared cache's symlinks.
The model is distributed under the [OpenMDW 1.1 license](https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX/blob/main/README.md).


The default dependencies are intended for macOS, Linux, and Windows.
Linux and Windows inference have not been runtime-tested.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and FFmpeg
(see below). From this checkout, install the command once:

```sh
uv tool install .
```

Then just give it a filename:

```sh
rek "Voice Memo.m4a"
```

No model, language, device, speaker-count, or tuning flags are required. The command
downloads the models if needed, decodes and normalizes the audio, detects speakers,
transcribes speech, and saves both text and JSON in a new `recordings/<timestamp>/`
directory under your current working directory. It prints the transcript and output
location when finished. First use needs internet access for the model downloads.
If your shell cannot find `rek` after installation, run `uv tool update-shell`
and reopen your terminal.

You can also run directly from the checkout without installing the command:

```sh
uv run main.py "Voice Memo.m4a"
```

The defaults are:

| Setting | Default |
| --- | --- |
| Hardware | Core ML on macOS; CPU on Linux/Windows |
| Speech recognition | English Parakeet v2 INT8, greedy decoding |
| Speaker detection | Nemotron 3 FP32, up to eight speakers automatically |
| Diarization windows | Published offline settings: 27.2 s + 3.2 s look-ahead, persistent speaker cache |
| Speaker activity | Probability above 0.5, at 10 ms resolution |
| Transcription windows | 30 s with 2 s context on each side |
| Audio | Automatic mono mixing and resampling to 16 kHz |
| Output | Timestamped text and JSON; existing output folders are never overwritten |

These use the model's published settings and bounded inference windows; they are
starting defaults, not benchmark-tuned settings for a particular machine.

Run `rek` without a filename to record your default microphone until **Ctrl+C**.
Start speaking when you see **“Recording”**. Ctrl+C finishes saving the WAV and starts
transcription. There is no duration limit or `--seconds` option.

The project uses Python 3.14; uv can install it automatically. Built-in model revisions
are pinned. Later runs reuse the Hugging Face cache, and audio stays on your computer.
To relocate the cache, set `HF_HOME`. For offline use, populate the cache first,
then set `HF_HUB_OFFLINE=1`.

Overrides are optional, mainly for troubleshooting or changing the destination:

```sh
rek meeting.wav --output-dir recordings/meeting
rek meeting.wav --device cpu
rek --help
```

The older `--input filename` syntax remains supported.

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

## Optional MLX diarization

On an Apple Silicon Mac, install the extra and select MLX explicitly:

```sh
uv run --extra mlx main.py "Voice Memo.m4a" --diarization-backend mlx
```

For an installed command, use `uv tool install '.[mlx]'` from this checkout,
then `rek meeting.wav --diarization-backend mlx`.
ONNX remains the default and Parakeet transcription still uses sherpa-onnx.
With MLX selected, `--device auto` uses the Apple GPU for diarization and requests
Core ML for ASR; `--device cpu` requests CPU for both.

The optional `mlx-audio` backend loads
[`mlx-community/Nemotron-3-Diarization`](https://huggingface.co/mlx-community/Nemotron-3-Diarization)
at revision `59ed2dbfc1346dcea9d423c71306a3a2499c568f` with strict weight loading.
It uses the offline preset (27.2 s chunks, 3.2 s look-ahead), fresh speaker state
per recording, and the same 0.5 threshold and overlap-preserving segment conversion
as ONNX. Feature extraction and cache implementations still differ; output parity
is not guaranteed. The JSON records the selected model, weights file, backend,
and requested diarization device separately from the ASR provider.

Compare diarization on your own recording:

```sh
uv run --extra mlx python -m rek.benchmark meeting.wav --output benchmark.json
```

The benchmark runs ONNX CPU, ONNX Core ML (macOS), and MLX GPU (Apple Silicon)
in separate processes. It measures model loading, first inference, median warm
inference (three repetitions by default), real-time factor, and peak process RSS.
A real-time factor of 0.1 means ten seconds of processing per hundred seconds of
audio. Audio decoding is excluded. `--repeats` and `--timeout` (seconds per backend,
default 600) are configurable. Existing report files are refused.

Download time is included in loading when weights are uncached; run again after
the first download for a useful startup comparison. RSS includes the whole process
and is not a dedicated GPU-memory measurement. Core ML may execute some operations
on CPU. Failures/timeouts are saved alongside successful results. Speaker turns
are included for inspection, but accuracy cannot be measured without labelled
reference segments. This benchmark does not measure transcription speed.

A local run on Apple M4 (2026-11-13), using one 22m56s recording and two warm
repetitions, measured:

| Diarization backend | Warm median | Peak process RSS |
| --- | ---: | ---: |
| ONNX CPU | 11.27 s | 1,675 MiB |
| ONNX Core ML | 26.31 s | 4,554 MiB |
| MLX GPU | 3.55 s | 687 MiB |

MLX was approximately 3.2× faster than ONNX CPU and 7.4× faster than ONNX Core ML
on this recording. ONNX reported Core ML support for only 138 of 749 graph nodes
across 100 partitions. All runs detected two speakers, with different segment
boundaries; no labelled accuracy evaluation was performed. These are single-machine,
single-recording results, not general speed guarantees. The run used ONNX Runtime
1.31.1, MLX 0.32.4 and mlx-audio 0.6.5. ONNX remains the default.

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

With the default ONNX backend, `--device auto` requests Core ML on macOS for both
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
fails or acceleration performs poorly. Use the benchmark command above to compare
the backends on your hardware.

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
ALAC files and are skipped when FFmpeg is unavailable. Tests cover both ONNX and
the optional MLX adapter. Ruff enforces annotations and a
complexity limit of 10, Pyright checks types, and source files are limited to 500 lines.

## Publishing

Version tags publish the `rek` distribution through GitHub Actions OIDC. No PyPI API
token or password is stored in GitHub. Configure a PyPI Trusted Publisher with:

| Field | Value |
| --- | --- |
| PyPI project | `rek` |
| GitHub owner | `gardner` |
| GitHub repository | `rek` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` |

Create the `pypi` environment in the repository settings. For the first upload,
register a pending publisher in your PyPI account's Publishing settings. See the
[PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

To release, update `version` in `pyproject.toml`, run `uv lock`, and commit and push
the changes. Create and push a matching version tag:

```sh
git tag -a v0.1.1 -m "Release rek 0.1.1"
git push origin v0.1.1
```

This example assumes `version = "0.1.1"` has already been committed.
The workflow checks that the tag matches the package version, builds a wheel and
source distribution, and publishes them from a separate job with `id-token: write`.
Pushing a `v*` tag triggers publishing automatically; no GitHub release is required.
Ordinary branch pushes do not publish. The tag must point to a commit containing
this tag-triggered workflow; tags pushed before that change are not retriggered.
Each new PyPI release needs a new version. The workflow does not run tests.

Once published, users can install and run the command with:

```sh
uv tool install rek
rek "Voice Memo.m4a"
```
