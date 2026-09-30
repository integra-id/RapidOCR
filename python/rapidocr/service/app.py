# -*- encoding: utf-8 -*-
"""HTTP API for Indonesian KTP OCR.

The process loads the ONNX models once at startup. Point
``RAPIDOCR_MODEL_DIR`` at a directory that already contains those files so a
request never has to download them.
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

from rapidocr.postprocess import KTP_FIELDS, parse_ktp
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
        "KTP",
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


def _lines_from_result(result) -> list[dict[str, Any]]:
    texts = getattr(result, "txts", None) or ()
    scores = getattr(result, "scores", None) or ()
    boxes = getattr(result, "boxes", None)
    lines = []
    for index, text in enumerate(texts):
        box = None
        if boxes is not None and index < len(boxes):
            box = boxes[index].tolist()
        score = float(scores[index]) if index < len(scores) else None
        lines.append({"text": text, "score": score, "box": box})
    return lines


def _elapse(result) -> Optional[float]:
    value = getattr(result, "elapse", None)
    if value is None:
        return None
    return float(value)


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

    app = FastAPI(title="RapidOCR KTP", version="1", lifespan=lifespan)

    @app.get("/health")
    def health():
        ready = bool(getattr(app.state, "ready", False))
        return {
            "status": "ok" if ready else "starting",
            "lang": "id",
            "models_ready": ready,
        }

    @app.post("/ocr")
    async def ocr(file: UploadFile = File(...)):
        result = _run(app, await _read_upload(file))
        return {"lines": _lines_from_result(result), "elapse": _elapse(result)}

    @app.post("/ktp")
    async def ktp(file: UploadFile = File(...)):
        payload = await _read_upload(file)
        result = _run(app, payload)
        fields = parse_ktp(result)
        return {
            "lines": _lines_from_result(result),
            "fields": {key: fields.get(key) for key in KTP_FIELDS},
            "elapse": _elapse(result),
        }

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
