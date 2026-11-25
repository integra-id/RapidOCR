# -*- encoding: utf-8 -*-
"""Runtime settings for the rapidocr-id HTTP service.

``RAPIDOCR_PROVIDER=cpu`` (default) stays on CPU. ``cuda`` or ``gpu`` asks
for CUDA and falls back to CPU when that provider is not installed or the
machine has no GPU.

``RAPIDOCR_WORKERS`` is the number of in-process OCR engine sessions
(default: half the CPU count, clamped to 2–4, or 1 on a single core).
``RAPIDOCR_MAX_SIDE`` caps the longest image side in pixels before OCR
(default 1280; ``0`` leaves the library default of 2000).
``RAPIDOCR_MIN_SHORT`` (default 960) is a short-side floor: a skinny
photo is not crushed below that height/width, even if the long side
then exceeds ``RAPIDOCR_MAX_SIDE``. Set it to ``0`` for a hard long-side
cap. Intra-op ONNX threads per session are ``max(1, cores // workers)``
unless ``RAPIDOCR_INTRA_THREADS`` is set, so workers × threads stay
near the core count.
"""

from __future__ import annotations

import os
from typing import Optional

from rapidocr.utils.log import logger

DEFAULT_MAX_SIDE = 1280
DEFAULT_MIN_SHORT = 960
LIBRARY_MAX_SIDE = 2000
ENGINE_SIDE_CEILING = 8192
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


def min_short() -> int:
    """Minimum short side in pixels after downscale. 0 disables the floor."""
    value = _int_env("RAPIDOCR_MIN_SHORT")
    if value is None:
        return DEFAULT_MIN_SHORT
    if value <= 0:
        return 0
    return max(32, min(value, ENGINE_SIDE_CEILING))


def target_max_side(height: int, width: int, cap: Optional[int] = None, floor: Optional[int] = None) -> int:
    """Long-side cap to use for this image, loosened to keep the short side."""
    cap = effective_max_side() if cap is None else cap
    floor = min_short() if floor is None else floor
    if cap <= 0:
        return 0
    long_side = max(int(height), int(width))
    short_side = min(int(height), int(width))
    if long_side <= cap:
        return 0
    if floor and short_side * cap / long_side < floor:
        return min(long_side, max(cap, int(round(floor * long_side / float(short_side)))))
    return cap


def engine_max_side() -> int:
    """Library ceiling so a min-short loosening is not undone inside the engine."""
    cap = max_side()
    if cap <= 0:
        return LIBRARY_MAX_SIDE
    if min_short() <= 0:
        return cap
    return max(cap, ENGINE_SIDE_CEILING)


def engine_params() -> dict:
    params: dict = {}
    cap = engine_max_side()
    if cap > 0:
        params["Global.max_side_len"] = cap
    if requested_provider() == "cuda":
        params["EngineConfig.onnxruntime.use_cuda"] = True
    else:
        params["EngineConfig.onnxruntime.intra_op_num_threads"] = intra_op_threads()
        params["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
    return params
