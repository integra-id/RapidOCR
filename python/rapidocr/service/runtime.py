# -*- encoding: utf-8 -*-
"""Runtime settings for the rapidocr-id HTTP service.

``RAPIDOCR_PROVIDER=cpu`` (default) stays on CPU. ``cuda`` or ``gpu`` asks
for CUDA and falls back to CPU when that provider is not installed or the
machine has no GPU.

``RAPIDOCR_WORKERS`` is the number of in-process OCR engine sessions
(default: half the CPU count, clamped to 2–4, or 1 on a single core).
``RAPIDOCR_MAX_SIDE`` caps the longest image side in pixels before OCR
(default 1280; ``0`` leaves the library default of 2000). Intra-op ONNX
threads per session are ``max(1, cores // workers)`` unless
``RAPIDOCR_INTRA_THREADS`` is set, so workers × threads stay near the
core count.
"""

from __future__ import annotations

import os
from typing import Optional

from rapidocr.utils.log import logger

DEFAULT_MAX_SIDE = 1280
LIBRARY_MAX_SIDE = 2000
MAX_WORKERS_CAP = 16


def cpu_count() -> int:
    return os.cpu_count() or 2


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


def _int_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        logger.warning("%s=%r is not an integer; using the default.", name, raw)
        return None


def default_worker_count(cores: Optional[int] = None) -> int:
    """Pick 1–4 workers from the CPU count when RAPIDOCR_WORKERS is unset."""
    n = cores if cores is not None else cpu_count()
    n = max(1, int(n))
    if n <= 1:
        return 1
    if n == 2:
        return 2
    return max(2, min(4, n // 2))


def worker_count() -> int:
    value = _int_env("RAPIDOCR_WORKERS")
    if value is None:
        return default_worker_count()
    return max(1, min(value, MAX_WORKERS_CAP))


def max_side() -> int:
    """Longest side cap in pixels. 0 keeps the RapidOCR library default."""
    value = _int_env("RAPIDOCR_MAX_SIDE")
    if value is None:
        return DEFAULT_MAX_SIDE
    if value <= 0:
        return 0
    return max(32, min(value, 8192))


def intra_op_threads(workers: Optional[int] = None, cores: Optional[int] = None) -> int:
    value = _int_env("RAPIDOCR_INTRA_THREADS")
    if value is not None:
        return max(1, min(value, 32))
    workers = workers if workers is not None else worker_count()
    cores = cores if cores is not None else cpu_count()
    return max(1, int(cores) // max(1, int(workers)))


def effective_max_side() -> int:
    cap = max_side()
    return LIBRARY_MAX_SIDE if cap == 0 else cap


def engine_params() -> dict:
    params: dict = {}
    cap = max_side()
    if cap > 0:
        params["Global.max_side_len"] = cap
    if requested_provider() == "cuda":
        params["EngineConfig.onnxruntime.use_cuda"] = True
    else:
        params["EngineConfig.onnxruntime.intra_op_num_threads"] = intra_op_threads()
        params["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
    return params
