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
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, create_model

from rapidocr.postprocess import KTP_FIELDS, parse_ktp
from rapidocr.service.pdf import (
    MEDIA_TYPES,
    DocumentError,
    collect_formats,
    looks_like_pdf,
    page_markdown,
    page_plain_text,
    rasterize_pdf,
    render_html,
    render_markdown,
    render_text,
)
from rapidocr.service.version import SERVICE_NAME, SERVICE_VERSION
from rapidocr.utils.load_image import LoadImageError
from rapidocr.utils.log import logger

DEFAULT_MODEL_DIR = "/opt/rapidocr/models"
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_PDF_BYTES = 20 * 1024 * 1024
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


def document_body(results) -> dict[str, Any]:
    pages = []
    elapsed = 0.0
    saw_elapse = False
    for index, result in enumerate(results, start=1):
        page_elapse = _elapse(result)
        if page_elapse is not None:
            saw_elapse = True
            elapsed += page_elapse
        lines = _ocr_lines(result)
        pages.append(
            {
                "page": index,
                "elapse": page_elapse,
                "lines": lines,
                "text": page_plain_text(lines),
                "markdown": page_markdown(result),
            }
        )
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "page_count": len(pages),
        "elapse": elapsed if saw_elapse else None,
        "text": "\n\n".join(page["text"] for page in pages if page["text"]),
        "pages": pages,
    }


def public_json_document(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "service": body["service"],
        "version": body["version"],
        "page_count": body["page_count"],
        "elapse": body["elapse"],
        "text": body["text"],
        "pages": [
            {
                "page": page["page"],
                "elapse": page["elapse"],
                "text": page["text"],
                "lines": page["lines"],
            }
            for page in body["pages"]
        ],
    }


def _rendered_output(body: dict[str, Any], export_format: str):
    if export_format == "json":
        return public_json_document(body)
    if export_format == "txt":
        return render_text(body["pages"])
    if export_format == "markdown":
        return render_markdown(body["pages"])
    return render_html(body["pages"])


def render_document(body: dict[str, Any], export_formats: list[str]):
    if len(export_formats) == 1:
        export_format = export_formats[0]
        rendered = _rendered_output(body, export_format)
        if export_format == "json":
            return JSONResponse(content=rendered, media_type=MEDIA_TYPES["json"])
        return Response(content=rendered, media_type=MEDIA_TYPES[export_format])

    return JSONResponse(
        content={
            "service": body["service"],
            "version": body["version"],
            "page_count": body["page_count"],
            "elapse": body["elapse"],
            "outputs": {
                export_format: _rendered_output(body, export_format)
                for export_format in export_formats
            },
        },
        media_type="application/json",
    )


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


async def _read_upload(file: UploadFile, limit: int) -> bytes:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    if len(payload) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Upload is larger than {limit // (1024 * 1024)} MB.",
        )
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
        responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    )
    async def ocr(
        file: UploadFile = File(...),
        formats: Optional[list[str]] = Query(None),
        format: Optional[list[str]] = Query(None),
        output: Optional[list[str]] = Query(None),
        formats_form: Optional[list[str]] = Form(None, alias="formats"),
        format_form: Optional[list[str]] = Form(None, alias="format"),
        output_form: Optional[list[str]] = Form(None, alias="output"),
    ):
        return await _recognize_upload(
            app,
            file,
            formats,
            format,
            output,
            formats_form,
            format_form,
            output_form,
            pdf_only=False,
        )

    @app.post(
        "/ocr/pdf",
        responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    )
    async def ocr_pdf(
        file: UploadFile = File(...),
        formats: Optional[list[str]] = Query(None),
        format: Optional[list[str]] = Query(None),
        output: Optional[list[str]] = Query(None),
        formats_form: Optional[list[str]] = Form(None, alias="formats"),
        format_form: Optional[list[str]] = Form(None, alias="format"),
        output_form: Optional[list[str]] = Form(None, alias="output"),
    ):
        return await _recognize_upload(
            app,
            file,
            formats,
            format,
            output,
            formats_form,
            format_form,
            output_form,
            pdf_only=True,
        )

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
        payload = await _read_upload(file, MAX_IMAGE_BYTES)
        if looks_like_pdf(payload, file.filename, file.content_type):
            raise HTTPException(
                status_code=400,
                detail="KTP parsing accepts an image, not a PDF.",
            )
        return parse_ktp_body(_run(app, payload))

    return app


async def _recognize_upload(
    app,
    file,
    formats,
    format,
    output,
    formats_form,
    format_form,
    output_form,
    pdf_only: bool,
):
    try:
        export_formats = collect_formats(
            formats, format, output, formats_form, format_form, output_form
        )
    except DocumentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    head = await file.read(5)
    rest = await file.read()
    payload = head + rest
    if not payload:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    pdf = looks_like_pdf(payload, file.filename, file.content_type)
    limit = MAX_PDF_BYTES if pdf else MAX_IMAGE_BYTES
    if len(payload) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Upload is larger than {limit // (1024 * 1024)} MB.",
        )
    if pdf_only and not pdf:
        raise HTTPException(status_code=400, detail="Upload is not a PDF.")
    if pdf:
        return _recognize_pdf(app, payload, export_formats)
    result = _run(app, payload)
    if export_formats == ["json"]:
        return OcrResponse(**ocr_body(result))
    return render_document(document_body([result]), export_formats)


def _recognize_pdf(app, payload: bytes, export_formats: list[str]):
    try:
        images = rasterize_pdf(payload)
    except DocumentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    results = _run_many(app, images)
    return render_document(document_body(results), export_formats)


def _engine(app: FastAPI):
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="OCR engine is not ready.")
    return engine


def _run(app: FastAPI, payload):
    try:
        with _infer_lock:
            return _engine(app)(payload)
    except LoadImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _run_many(app: FastAPI, images):
    try:
        with _infer_lock:
            engine = _engine(app)
            return [engine(image) for image in images]
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
