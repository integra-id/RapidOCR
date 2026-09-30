# -*- encoding: utf-8 -*-
"""HTTP API for the rapidocr-id service.

General Indonesian OCR is the primary API (``/ocr``, ``/health``). Document
parsers such as KTP live under ``/parse``. The process loads ONNX models at
startup from ``RAPIDOCR_MODEL_DIR``. The production image downloads those
files while it is built, so a request does not fetch them.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, create_model

from rapidocr.postprocess import KTP_FIELDS, parse_ktp
from rapidocr.service.version import SERVICE_NAME, SERVICE_VERSION
from rapidocr.utils.load_image import LoadImageError
from rapidocr.utils.log import logger

DEFAULT_MODEL_DIR = "/opt/rapidocr/models"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
_infer_lock = threading.Lock()


def model_dir() -> Path:
    return Path(os.environ.get("RAPIDOCR_MODEL_DIR", DEFAULT_MODEL_DIR))


def preload():
    """Create RapidOCR and load det, cls, and rec sessions.

    Detection on a blank image returns no crops, so cls and rec would stay
    unloaded. Loading each session here downloads any missing file once and
    keeps later requests on local disk.
    """
    from rapidocr import RapidOCR

    destination = model_dir()
    destination.mkdir(parents=True, exist_ok=True)
    engine = RapidOCR(params={"Global.model_root_dir": str(destination)})
    engine._load_det_model()
    engine._load_cls_model()
    engine._load_rec_model()
    _touch_pipeline(engine)
    logger.info("RapidOCR ready with models in %s", destination)
    return engine


def _touch_pipeline(engine) -> None:
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
        logger.info("Warmup image produced no text; model sessions are loaded.")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    lang: str
    models_ready: bool


class VersionResponse(BaseModel):
    service: str
    version: str


class ErrorResponse(BaseModel):
    detail: str


class OcrLine(BaseModel):
    text: str
    score: Optional[float] = None
    box: Optional[list[list[float]]] = None


class OcrResponse(BaseModel):
    service: str
    version: str
    elapse: Optional[float] = None
    lines: list[OcrLine]


class KtpLine(BaseModel):
    """One recognized line. KTP parsing does not use the box."""

    text: str
    score: Optional[float] = None


KtpFields = create_model(
    "KtpFields",
    __config__=ConfigDict(extra="ignore"),
    **{key: (Optional[str], None) for key in KTP_FIELDS},
)


class ParseKtpResponse(BaseModel):
    service: str
    version: str
    elapse: Optional[float] = None
    lines: list[KtpLine]
    fields: KtpFields


def _texts(result) -> list[str]:
    raw = getattr(result, "txts", None) or ()
    return ["" if text is None else str(text) for text in raw]


def _score(result, index: int) -> Optional[float]:
    scores = getattr(result, "scores", None) or ()
    if index >= len(scores) or scores[index] is None:
        return None
    return float(scores[index])


def _box(result, index: int) -> Optional[list[list[float]]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or index >= len(boxes) or boxes[index] is None:
        return None
    return [[float(point) for point in corner] for corner in boxes[index].tolist()]


def _ocr_lines(result) -> list[dict[str, Any]]:
    return [
        {"text": text, "score": _score(result, index), "box": _box(result, index)}
        for index, text in enumerate(_texts(result))
    ]


def _ktp_lines(result) -> list[dict[str, Any]]:
    return [
        {"text": text, "score": _score(result, index)}
        for index, text in enumerate(_texts(result))
    ]


def _elapse(result) -> Optional[float]:
    value = getattr(result, "elapse", None)
    if value is None:
        return None
    return float(value)


def ocr_body(result) -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "elapse": _elapse(result),
        "lines": _ocr_lines(result),
    }


def parse_ktp_body(result) -> dict[str, Any]:
    """Parse KTP from recognition strings only.

    Phone-photo boxes are not in reading order. Passing them into
    ``parse_ktp`` reorders lines and breaks field mapping.
    """
    fields = parse_ktp(_texts(result))
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "elapse": _elapse(result),
        "lines": _ktp_lines(result),
        "fields": {key: fields.get(key) for key in KTP_FIELDS},
    }


async def _read_upload(file: UploadFile) -> bytes:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 15 MB.")
    return payload


def create_app(loader: Callable[[], Any] = preload) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = loader()
        app.state.ready = True
        yield
        app.state.ready = False

    app = FastAPI(
        title=SERVICE_NAME,
        version=SERVICE_VERSION,
        lifespan=lifespan,
        default_response_class=JSONResponse,
    )

    @app.exception_handler(HTTPException)
    async def http_error(_request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": detail},
            media_type="application/json",
        )

    @app.get("/health", response_model=HealthResponse)
    def health():
        ready = bool(getattr(app.state, "ready", False))
        return {
            "status": "ok" if ready else "starting",
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "lang": "id",
            "models_ready": ready,
        }

    @app.get("/version", response_model=VersionResponse)
    def version():
        return {"service": SERVICE_NAME, "version": SERVICE_VERSION}

    @app.post(
        "/ocr",
        response_model=OcrResponse,
        responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    )
    async def ocr(file: UploadFile = File(...)):
        return ocr_body(_run(app, await _read_upload(file)))

    @app.post(
        "/parse/ktp",
        response_model=ParseKtpResponse,
        tags=["parsers"],
        responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    )
    @app.post(
        "/ktp",
        response_model=ParseKtpResponse,
        tags=["parsers"],
        include_in_schema=False,
    )
    async def parse_ktp_document(file: UploadFile = File(...)):
        return parse_ktp_body(_run(app, await _read_upload(file)))

    return app


def _run(app: FastAPI, payload: bytes):
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="OCR engine is not ready.")
    try:
        with _infer_lock:
            return engine(payload)
    except LoadImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "rapidocr.service.app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
