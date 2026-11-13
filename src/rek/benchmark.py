"""Compare diarization runtimes in separate processes on the same recording."""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Literal

from rek.audio import SAMPLE_RATE, load_audio
from rek.transcript import SpeakerTurn


def peak_rss_mib() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def measure(
    path: Path, backend: str, device: Literal["cpu", "coreml"], repeats: int
) -> dict[str, object]:
    """Exclude audio decoding; separate model loading, first inference and warm runs."""
    audio = load_audio(path)
    duration = len(audio) / SAMPLE_RATE
    if duration < 0.01:
        raise ValueError("Benchmark audio must contain at least 10 ms.")
    start = perf_counter()
    if backend == "mlx":
        from rek.mlx_diarization import load_diarizer
    else:
        from rek.diarization import load_diarizer
    infer = load_diarizer(device)
    load_seconds = perf_counter() - start
    times: list[float] = []
    turns: list[SpeakerTurn] = []
    for _ in range(repeats + 1):
        start = perf_counter()
        turns = infer(audio)
        times.append(perf_counter() - start)
    warm = statistics.median(times[1:])
    return {
        "backend": backend,
        "requested_device": "gpu" if backend == "mlx" and device == "coreml" else device,
        "audio_seconds": duration,
        "load_seconds_including_uncached_downloads": load_seconds,
        "first_inference_seconds": times[0],
        "warm_inference_seconds": times[1:],
        "warm_median_seconds": warm,
        "warm_real_time_factor": warm / duration,
        "peak_process_rss_mib": peak_rss_mib(),
        "speaker_turns": [asdict(turn) for turn in turns],
    }


def package_versions() -> dict[str, str]:
    packages = {}
    for name in ("onnxruntime", "mlx", "mlx-audio", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return packages


def compare(path: Path, repeats: int, timeout: float) -> dict[str, object]:
    candidates = [("onnx", "cpu")]
    if sys.platform == "darwin":
        candidates.append(("onnx", "coreml"))
        if platform.machine() == "arm64":
            candidates.append(("mlx", "coreml"))
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="rek-benchmark-") as directory:
        for backend, device in candidates:
            print(f"Benchmarking {backend}/{device}...", flush=True)
            output = Path(directory) / f"{backend}-{device}.json"
            command = [
                sys.executable,
                "-m",
                "rek.benchmark",
                str(path.resolve()),
                "--worker",
                backend,
                "--device",
                device,
                "--repeats",
                str(repeats),
                "--output",
                str(output),
            ]
            try:
                process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
                if process.returncode:
                    results.append(
                        {
                            "backend": backend,
                            "requested_device": device,
                            "error": process.stderr[-4000:],
                        }
                    )
                else:
                    result = json.loads(output.read_text())
                    result["runtime_messages"] = process.stderr[-4000:]
                    results.append(result)
                    print(f"  warm median: {result['warm_median_seconds']:.3f}s", flush=True)
            except subprocess.TimeoutExpired:
                results.append(
                    {
                        "backend": backend,
                        "requested_device": device,
                        "error": f"Timed out after {timeout:g} seconds",
                    }
                )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "audio_file": str(path.resolve()),
        "settings": {"chunk_seconds": 27.2, "lookahead_seconds": 3.2, "threshold": 0.5},
        "notes": [
            "Fresh process per backend; warm runs reuse the model but reset speaker state.",
            "Loading includes downloads; rerun with cached weights for startup comparison.",
            "RSS is whole-process peak memory, not GPU allocation or model weight size.",
            "Core ML may partition operations onto CPU; requested device is not measured usage.",
            "Speaker turns are saved for review. Accuracy requires labelled reference audio.",
        ],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--worker", choices=["onnx", "mlx"], help=argparse.SUPPRESS)
    parser.add_argument(
        "--device", choices=["cpu", "coreml"], default="cpu", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new filename")
    result = (
        measure(args.audio, args.worker, args.device, args.repeats)
        if args.worker
        else compare(args.audio, args.repeats, args.timeout)
    )
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
