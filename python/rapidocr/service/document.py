# -*- encoding: utf-8 -*-
"""Opt-in document OCR profile for rapidocr-id.

This module must not change the default / KTP engine. Existing environment
variables (``RAPIDOCR_MAX_SIDE``, ``RAPIDOCR_MIN_SHORT``, ``RAPIDOCR_WORKERS``,
and the rest) keep their current meaning. Document settings use the
``RAPIDOCR_DOC_*`` prefix and are read only when a client opts in through
``profile=document``, ``POST /ocr/document``, or ``task=document``.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Callable, Optional

from rapidocr.service.pdf import DocumentError
from rapidocr.service.runtime import (
    LIBRARY_MAX_SIDE,
    MAX_WORKERS_CAP,
    _int_env,
    active_provider,
    cpu_count,
    requested_provider,
)
from rapidocr.utils.log import logger

PROFILE_DEFAULT = "default"
PROFILE_DOCUMENT = "document"

DEFAULT_DOC_DPI = 200
DEFAULT_DOC_MAX_SIDE = 0
DEFAULT_DOC_MIN_SHORT = 0
DEFAULT_DOC_ENGINE_MAX_SIDE = LIBRARY_MAX_SIDE  # 2000; 3000+ was worse on 200 DPI A4 scans
DEFAULT_DOC_MODEL = "tiny"
DEFAULT_DOC_WORKERS = 1

# Shared by the first document request (lazy) and optional startup preload.
_document_pool_lock = threading.Lock()


def parse_profile(value: Optional[str], default: str = PROFILE_DEFAULT) -> str:
    """Return ``default`` or ``document``. Unknown values raise DocumentError."""
    if value is None or str(value).strip() == "":
        return default
    token = str(value).strip().lower()
    if token in {"default", "ktp", "standard"}:
        return PROFILE_DEFAULT
    if token in {"document", "doc", "documents"}:
        return PROFILE_DOCUMENT
    raise DocumentError("profile must be default or document.")


def parse_document_model(value: Optional[str]) -> str:
    """Return ``tiny`` or ``small``. Empty uses ``RAPIDOCR_DOC_MODEL``."""
    if value is None or str(value).strip() == "":
        return document_model_name()
    token = str(value).strip().lower()
    if token in {"tiny", "small"}:
        return token
    raise DocumentError("model must be tiny or small.")


def parse_fix_digits(value: Optional[str], default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise DocumentError("fix_digits must be true or false.")


def document_model_name() -> str:
    raw = os.environ.get("RAPIDOCR_DOC_MODEL", DEFAULT_DOC_MODEL)
    token = (raw or DEFAULT_DOC_MODEL).strip().lower()
    if token not in {"tiny", "small"}:
        logger.warning("RAPIDOCR_DOC_MODEL=%r is invalid; using %s.", raw, DEFAULT_DOC_MODEL)
        return DEFAULT_DOC_MODEL
    return token


def document_dpi() -> float:
    value = _int_env("RAPIDOCR_DOC_DPI")
    if value is None:
        return float(DEFAULT_DOC_DPI)
    if value < 72 or value > 300:
        logger.warning("RAPIDOCR_DOC_DPI=%s is out of 72–300; using %s.", value, DEFAULT_DOC_DPI)
        return float(DEFAULT_DOC_DPI)
    return float(value)


def document_max_side() -> int:
    """Service long-side cap for the document profile. Default 0 = no cap."""
    value = _int_env("RAPIDOCR_DOC_MAX_SIDE")
    if value is None:
        return DEFAULT_DOC_MAX_SIDE
    if value <= 0:
        return 0
    return max(32, min(value, 8192))


def document_min_short() -> int:
    value = _int_env("RAPIDOCR_DOC_MIN_SHORT")
    if value is None:
        return DEFAULT_DOC_MIN_SHORT
    if value <= 0:
        return 0
    return max(32, min(value, 8192))


def document_engine_max_side() -> int:
    """RapidOCR Global.max_side_len for the document engine (library default 2000)."""
    value = _int_env("RAPIDOCR_DOC_ENGINE_MAX_SIDE")
    if value is None:
        return DEFAULT_DOC_ENGINE_MAX_SIDE
    if value <= 0:
        return LIBRARY_MAX_SIDE
    return max(32, min(value, 8192))


def document_worker_count() -> int:
    value = _int_env("RAPIDOCR_DOC_WORKERS")
    if value is None:
        return DEFAULT_DOC_WORKERS
    return max(1, min(value, MAX_WORKERS_CAP))


def document_intra_op_threads(workers: Optional[int] = None, cores: Optional[int] = None) -> int:
    value = _int_env("RAPIDOCR_DOC_INTRA_THREADS")
    if value is not None:
        return max(1, min(value, 32))
    workers = workers if workers is not None else document_worker_count()
    cores = cores if cores is not None else cpu_count()
    return max(1, int(cores) // max(1, int(workers)))


def document_preload_enabled() -> bool:
    raw = os.environ.get("RAPIDOCR_DOC_PRELOAD", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def document_engine_params(model: Optional[str] = None) -> dict:
    """Params for a document RapidOCR engine. Independent of RAPIDOCR_MAX_SIDE."""
    from rapidocr import ModelType, OCRVersion

    name = parse_document_model(model)
    model_type = ModelType.TINY if name == "tiny" else ModelType.SMALL
    params: dict = {
        "Global.use_cls": False,
        "Global.cls_score_flip_guard": False,
        "Global.max_side_len": document_engine_max_side(),
        "Det.model_type": model_type,
        "Rec.model_type": model_type,
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Rec.ocr_version": OCRVersion.PPOCRV6,
    }
    if requested_provider() == "cuda":
        params["EngineConfig.onnxruntime.use_cuda"] = True
    else:
        params["EngineConfig.onnxruntime.intra_op_num_threads"] = document_intra_op_threads()
        params["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
    return params


def _document_model_dir():
    from pathlib import Path

    return Path(os.environ.get("RAPIDOCR_MODEL_DIR", "/opt/rapidocr/models"))


def preload_document(model: Optional[str] = None):
    """Load the document engine (PP-OCRv6 tiny, cls off, by default)."""
    import cv2
    import numpy as np
    from rapidocr import RapidOCR
    from rapidocr.service.runtime import requested_provider

    destination = _document_model_dir()
    destination.mkdir(parents=True, exist_ok=True)
    params = {"Global.model_root_dir": str(destination)}
    params.update(document_engine_params(model))
    if requested_provider() == "cuda" and active_provider() != "cuda":
        logger.warning("RAPIDOCR_PROVIDER asks for CUDA. CPU will be used instead.")
    engine = RapidOCR(params=params)
    engine._load_det_model()
    # Classification stays unloaded: document profile keeps use_cls False.
    engine._load_rec_model()
    image = np.full((64, 320, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "OCR",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    try:
        engine(image)
    except Exception:
        logger.info("Document warmup image produced no text; model sessions are loaded.")
    logger.info(
        "Document OCR ready (%s, cls off, max_side_len=%s) with models in %s",
        parse_document_model(model),
        document_engine_max_side(),
        destination,
    )
    return engine


def get_or_create_document_pool(app, loader: Callable[[], Any], build_pool: Callable[..., Any]):
    """Lazy-create the document engine pool, once, under a lock."""
    pool = getattr(app.state, "document_pool", None)
    if pool is not None:
        return pool
    with _document_pool_lock:
        pool = getattr(app.state, "document_pool", None)
        if pool is not None:
            return pool
        pool = build_pool(loader, size=document_worker_count())
        app.state.document_pool = pool
        logger.info(
            "Document OCR pool ready: %s session(s), dpi=%s, max_side=%s, min_short=%s, engine_max_side=%s",
            pool.size,
            document_dpi(),
            document_max_side(),
            document_min_short(),
            document_engine_max_side(),
        )
        return pool


_DIGIT_GROUP_O = re.compile(r"(?<![\w])\d[\d.Oo]*[oO][\d.Oo]*\d(?![\w])")


def fix_o_in_digit_groups(text: str) -> str:
    """Replace o/O with 0 inside mostly-numeric groups like 275.0oo.000.

    Off by default. Enable with ``fix_digits=true`` on the document profile only.
    """

    def repl(match: re.Match) -> str:
        return re.sub(r"[oO]", "0", match.group(0))

    return _DIGIT_GROUP_O.sub(repl, text or "")


def apply_fix_digits_to_text(text: str, enabled: bool) -> str:
    if not enabled:
        return text
    return fix_o_in_digit_groups(text)


def apply_fix_digits_to_lines(lines: list[dict[str, Any]], enabled: bool) -> list[dict[str, Any]]:
    if not enabled:
        return lines
    fixed = []
    for line in lines:
        item = dict(line)
        if item.get("text"):
            item["text"] = fix_o_in_digit_groups(str(item["text"]))
        fixed.append(item)
    return fixed
