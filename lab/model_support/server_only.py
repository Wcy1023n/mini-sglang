"""Guards for model-support commands that must run on the GPU server."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def require_server_model_path() -> str:
    if platform.system() != "Linux":
        raise RuntimeError(
            "This command is server-only: run it on Linux with an NVIDIA CUDA GPU."
        )

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed in the server environment.") from error
    if not torch.cuda.is_available():
        raise RuntimeError("This command requires an NVIDIA CUDA GPU.")

    raw_path = os.environ.get("MODEL_PATH")
    if not raw_path:
        raise RuntimeError(
            "Set MODEL_PATH to the model directory already present on the server."
        )

    model_path = Path(raw_path).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"MODEL_PATH is not a directory: {model_path}")
    return str(model_path.resolve())
