# rek

[GitHub repository](https://github.com/gardner/rek) · [Issues](https://github.com/gardner/rek/issues)

Record your microphone or transcribe an audio file locally with timestamps and
speaker labels. **The backend is selected automatically:** MLX on Apple Silicon
Macs, ONNX on Windows, Linux, and Intel Macs. The default package includes ONNX
Runtime and sherpa-onnx on all platforms, plus MLX on Apple Silicon. No extras
are required; the platform is checked at runtime and only the selected inference
backend is loaded.

- [Parakeet TDT 0.6B v2 MLX](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2)
  transcribes English with aligned word timestamps. Revision:
  `8ae155301e23d820d82aa60d24817c900e69e487`.
- [Nemotron 3 Diarization MLX](https://huggingface.co/mlx-community/Nemotron-3-Diarization)
  detects up to eight anonymous speakers, including overlapping speech, at 10 ms
  resolution. Revision: `59ed2dbfc1346dcea9d423c71306a3a2499c568f`.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and FFmpeg
(`brew install ffmpeg` on macOS). From this checkout:

```sh
uv run main.py "Voice Memo.m4a"
```

Or install the command:

```sh
uv tool install .
rek "Voice Memo.m4a"
```

No backend or model flags are required. Run without a filename to record your
microphone until **Ctrl+C**, then transcribe. Models download automatically on first
use; subsequent runs reuse the Hugging Face cache. Audio stays on your computer.
The project uses Python 3.14, which uv can install automatically.

`--backend auto` (the default) uses MLX when Python runs natively on Apple Silicon
and ONNX elsewhere, including Intel Python under Rosetta. `--device auto` uses
the GPU with MLX, Core ML with ONNX on macOS, and CPU on Windows/Linux.
`--device cpu` forces CPU. Explicit `--backend mlx` still requires Apple Silicon.

M4A uses FFmpeg; WAV, FLAC, and other libsndfile formats are also supported. Audio
is mixed to mono and resampled to 16 kHz. The input file is never modified. The
legacy `--input filename` syntax remains supported. `--language en` is optional;
other languages are rejected because Parakeet v2 is English-only.

Nemotron uses its offline preset: 27.2-second chunks, 3.2 seconds of look-ahead,
and persistent speaker state within each recording. Speaker activity is thresholded
at 0.5, preserving overlaps. MLX Parakeet uses 30-second chunks with two seconds of
overlap and the library's token alignment/merge implementation. Transcription starts
after recording ends. No-speech input skips transcription.

## Backend selection

The default installation supports ONNX, including on Apple Silicon. To select it
explicitly:

```sh
uv run main.py meeting.wav --backend onnx
```

For an installed command:

```sh
rek meeting.wav --backend onnx
```

The ONNX backend selects the pinned Nemotron
FP32 ONNX export and English Parakeet v2 INT8 export. On macOS, auto requests
Core ML; elsewhere it uses CPU. `--device cpu` forces CPU. Core ML may partition
operations onto CPU and may be slower than CPU alone. CUDA is not configured.
Linux and Windows inference have not been runtime-tested.

`--diarization-backend` and `--asr-backend` override individual stages when needed;
otherwise both inherit `--backend` (default `auto`). `--asr-model` accepts a repository
or local directory compatible with the selected transcription backend. Its default
is chosen automatically for that backend. MLX is installed only on Apple Silicon;
an ONNX-only installation on other platforms does not pull MLX dependencies.
If ONNX dependencies are missing, the error provides instructions to repair the installation.

ONNX assets are pinned in `src/rek/model_files.py`. Nemotron's graph and external
weights are cached together as regular files because ONNX Runtime rejects external
weight paths through the shared cache's symlinks. MLX and ONNX use the same segment
thresholding, but precision, feature extraction, caching, and transcription decoding
differ; output parity is not guaranteed.

## Output

Each run creates a new `recordings/<timestamp>/` directory with:

- `recording.wav`: microphone audio, when recording.
- `transcript.txt`: timestamped speaker turns, also printed to the terminal.
- `transcript.json`: words, speaker turns, transcript, model IDs, selected backends,
  requested devices, and source audio path.

Use `--output-dir path` to select a new destination. Existing directories are
refused. A completed microphone recording survives inference failures or cancellation.
Words are assigned by greatest time overlap with speaker turns; ties and unmatched
words are labelled `Unknown`. Speaker IDs are anonymous within one recording.
Overlapping speech is detected but mixed voices are not separated.

Set `HF_HOME` to relocate model caches. Populate the cache before setting
`HF_HUB_OFFLINE=1` for offline use.

## Diarization benchmark

```sh
uv run python -m rek.benchmark meeting.wav --output benchmark.json
```

Each backend runs in a separate process. Reports separate loading (including any
uncached downloads), first inference, and median warm inference; decoding is excluded.
`--repeats` defaults to three warm runs and `--timeout` to 600 seconds per backend.
Reports include peak process RSS, runtime warnings, errors, and speaker turns.
Existing report files are refused. RSS is not a dedicated GPU-memory measurement;
accuracy requires labelled reference audio. The benchmark excludes transcription.

A local Apple M4 run on 2026-11-13, using one 22m56s recording and two warm runs:

| Diarization backend | Warm median | Peak process RSS |
| --- | ---: | ---: |
| MLX GPU | 3.55 s | 687 MiB |
| ONNX CPU | 11.27 s | 1,675 MiB |
| ONNX Core ML | 26.31 s | 4,554 MiB |

MLX was approximately 3.2× faster than ONNX CPU and 7.4× faster than ONNX Core ML
on this recording. All detected two speakers, with different boundaries; no labelled
accuracy evaluation was performed. This is one recording on one machine, not a
speed guarantee. Versions: ONNX Runtime 1.31.1, MLX 0.32.4, mlx-audio 0.6.5.

## Development

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run python scripts/check_file_length.py
```

Unit tests mock model inference and need no downloaded weights.
M4A tests use FFmpeg when available. Ruff enforces annotations and complexity <=10,
Pyright checks types, and source files are limited to 500 lines.

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
