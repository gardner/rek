"""Backend selection based on the running Python platform."""

import platform
import sys
from typing import Literal

type Backend = Literal["mlx", "onnx"]


def default_backend() -> Backend:
    """Use MLX only with native Apple Silicon Python, matching package markers."""
    return "mlx" if sys.platform == "darwin" and platform.machine() == "arm64" else "onnx"
