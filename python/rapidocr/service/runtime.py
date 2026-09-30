# -*- encoding: utf-8 -*-
"""Choose an ONNX Runtime provider for rapidocr-id.

``RAPIDOCR_PROVIDER=cpu`` (default) stays on CPU. ``cuda`` or ``gpu`` asks
for CUDA and falls back to CPU when that provider is not installed or the
machine has no GPU.
"""

from __future__ import annotations

import os

from rapidocr.utils.log import logger


def requested_provider() -> str:
    value = os.environ.get("RAPIDOCR_PROVIDER", "cpu").strip().lower()
    if value in {"cuda", "gpu"}:
        return "cuda"
    return "cpu"


def active_provider() -> str:
    if requested_provider() != "cuda":
        return "cpu"
    try:
        from onnxruntime import get_available_providers

        if "CUDAExecutionProvider" in get_available_providers():
            return "cuda"
    except Exception:
        logger.warning("CUDA was requested but ONNX Runtime could not be queried.")
    logger.warning("CUDA provider is unavailable. Falling back to CPU.")
    return "cpu"


def engine_params() -> dict:
    if requested_provider() == "cuda":
        return {"EngineConfig.onnxruntime.use_cuda": True}
    return {}
