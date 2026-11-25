# -*- encoding: utf-8 -*-
"""HTTP API for the rapidocr-id service.

General Indonesian OCR is the primary API (``/ocr``, ``/health``). Document
parsers such as KTP live under ``/parse``. The process loads ONNX models at
startup from ``RAPIDOCR_MODEL_DIR``. The production image downloads those
files while it is built, so a request does not fetch them.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, create_model

from rapidocr.postprocess import (
    INVOICE_FIELDS,
    KTP_FIELDS,
    NPWP_FIELDS,
    parse_invoice,
    parse_ktp,
    parse_npwp,
)
from rapidocr.postprocess.fuzzy_labels import compact as compact_text
from rapidocr.service.preprocess import apply_crop, parse_preprocess, preprocess_bgr, resize_max_side, scale_result_boxes
from rapidocr.service.artifacts import detect_artifacts
from rapidocr.service.auth import install_guard
from rapidocr.service.jobs import JobStore, expand_uploads, job_root
from rapidocr.service.layout import detect_tables, layout_html, render_csv
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
from rapidocr.service.runtime import (
    active_provider,
    effective_max_side,
    engine_params,
    max_side,
    min_short,
    requested_provider,
    worker_count,
)
from rapidocr.service.document import (
    PROFILE_DOCUMENT,
    apply_fix_digits_to_lines,
    apply_fix_digits_to_text,
    document_dpi,
    document_max_side,
    document_min_short,
    document_preload_enabled,
    get_or_create_document_pool,
    parse_document_model,
    parse_fix_digits,
    parse_profile,
    preload_document,
)
from rapidocr.service.pool import EnginePool, build_pool
from rapidocr.service.version import SERVICE_NAME, SERVICE_VERSION
from rapidocr.utils.load_image import LoadImage, LoadImageError
from rapidocr.utils.log import logger

DEFAULT_MODEL_DIR = "/opt/rapidocr/models"
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_PDF_BYTES = 20 * 1024 * 1024
DEFAULT_MIN_SCORE = 0.5
def requested_cuda_unavailable() -> bool:
    return requested_provider() == "cuda" and active_provider() != "cuda"


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
    params = {"Global.model_root_dir": str(destination)}
    params.update(engine_params())
    if requested_cuda_unavailable():
        logger.warning("RAPIDOCR_PROVIDER asks for CUDA. CPU will be used instead.")
    engine = RapidOCR(params=params)
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
    provider: str
    models_ready: bool
    workers: int
    max_side: int
    min_short: int


class VersionResponse(BaseModel):
    service: str
    version: str
    provider: str
    workers: int
    max_side: int
    min_short: int


class Artifact(BaseModel):
    type: str
    box: list[list[float]]
    score: float


class ErrorResponse(BaseModel):
    detail: str


class OcrLine(BaseModel):
    text: str
    score: Optional[float] = None
    box: Optional[list[list[float]]] = None
    below_min_score: bool = False


class OcrResponse(BaseModel):
    service: str
    version: str
    elapse: Optional[float] = None
    min_score: float = DEFAULT_MIN_SCORE
    lines: list[OcrLine]
    artifacts: list[Artifact] = []


class KtpLine(BaseModel):
    """One recognized line. Document parsers do not use the box."""

    text: str
    score: Optional[float] = None
    below_min_score: bool = False


KtpFields = create_model(
    "KtpFields",
    __config__=ConfigDict(extra="ignore"),
    **{key: (Optional[str], None) for key in KTP_FIELDS},
)


class ParseKtpResponse(BaseModel):
    service: str
    version: str
    elapse: Optional[float] = None
    min_score: float = DEFAULT_MIN_SCORE
    lines: list[KtpLine]
    fields: KtpFields
    field_scores: dict[str, Optional[float]]
    artifacts: list[Artifact] = []


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


def _mark_score(line: dict[str, Any], min_score: float) -> dict[str, Any]:
    score = line.get("score")
    line["below_min_score"] = score is not None and score < min_score
    return line


def _ocr_lines(result, min_score: float = DEFAULT_MIN_SCORE) -> list[dict[str, Any]]:
    return [
        _mark_score(
            {"text": text, "score": _score(result, index), "box": _box(result, index)},
            min_score,
        )
        for index, text in enumerate(_texts(result))
    ]


def _ktp_lines(result, min_score: float = DEFAULT_MIN_SCORE) -> list[dict[str, Any]]:
    return [
        _mark_score({"text": text, "score": _score(result, index)}, min_score)
        for index, text in enumerate(_texts(result))
    ]


def _elapse(result) -> Optional[float]:
    value = getattr(result, "elapse", None)
    if value is None:
        return None
    return float(value)


def ocr_body(
    result, min_score: float = DEFAULT_MIN_SCORE, artifacts: Optional[list] = None
) -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "elapse": _elapse(result),
        "min_score": min_score,
        "lines": _ocr_lines(result, min_score),
        "artifacts": artifacts or [],
    }


def document_body(
    results, min_score: float = DEFAULT_MIN_SCORE, images: Optional[list] = None
) -> dict[str, Any]:
    pages = []
    elapsed = 0.0
    saw_elapse = False
    for index, result in enumerate(results, start=1):
        page_elapse = _elapse(result)
        if page_elapse is not None:
            saw_elapse = True
            elapsed += page_elapse
        lines = _ocr_lines(result, min_score)
        pages.append(
            {
                "page": index,
                "elapse": page_elapse,
                "lines": lines,
                "text": page_plain_text(lines),
                "markdown": page_markdown(lines),
                "html_body": layout_html(lines),
                "tables": detect_tables(lines),
                "artifacts": _page_artifacts(images, index - 1, lines),
            }
        )
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "page_count": len(pages),
        "elapse": elapsed if saw_elapse else None,
        "min_score": min_score,
        "text": "\n\n".join(page["text"] for page in pages if page["text"]),
        "pages": pages,
    }


def public_json_document(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "service": body["service"],
        "version": body["version"],
        "page_count": body["page_count"],
        "elapse": body["elapse"],
        "min_score": body.get("min_score", DEFAULT_MIN_SCORE),
        "text": body["text"],
        "pages": [
            {
                "page": page["page"],
                "elapse": page["elapse"],
                "text": page["text"],
                "lines": page["lines"],
                "tables": page.get("tables") or [],
                "artifacts": page.get("artifacts") or [],
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
    if export_format == "csv":
        return render_csv(body["pages"])
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
            "min_score": body.get("min_score", DEFAULT_MIN_SCORE),
            "outputs": {
                export_format: _rendered_output(body, export_format)
                for export_format in export_formats
            },
        },
        media_type="application/json",
    )


def gate_fields(fields: dict[str, Optional[str]], lines, field_names, min_score: float):
    """Drop a field when the line that supplied it is below ``min_score``."""
    kept = {}
    scores = {}
    for key in field_names:
        value = fields.get(key)
        if not value:
            kept[key] = None
            scores[key] = None
            continue
        score = _matching_score(value, lines)
        scores[key] = score
        if score is not None and score < min_score:
            kept[key] = None
        else:
            kept[key] = value
    return kept, scores


def _matching_score(value: str, lines) -> Optional[float]:
    target = compact_text(value)
    if not target:
        return None
    best_score = None
    best_overlap = 0
    for line in lines:
        source = compact_text(line.get("text") or "")
        if not source:
            continue
        if target in source or source in target:
            overlap = min(len(target), len(source))
            if overlap > best_overlap:
                best_overlap = overlap
                best_score = line.get("score")
    return best_score


def _page_artifacts(images, index: int, lines) -> list:
    if not images or index >= len(images) or images[index] is None:
        return []
    return detect_artifacts(images[index], lines)


def parse_document_body(
    result, parser, field_names, min_score: float, artifacts: Optional[list] = None
) -> dict[str, Any]:
    """Parse from recognition strings only, ignoring box order."""
    lines = _ktp_lines(result, min_score)
    texts = [line["text"] for line in lines if not line["below_min_score"]]
    raw_fields = parser(texts)
    fields, scores = gate_fields(raw_fields, lines, field_names, min_score)
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "elapse": _elapse(result),
        "min_score": min_score,
        "lines": lines,
        "fields": fields,
        "field_scores": scores,
        "artifacts": artifacts or [],
    }


def parse_ktp_body(result, min_score: float = DEFAULT_MIN_SCORE) -> dict[str, Any]:
    return parse_document_body(result, parse_ktp, KTP_FIELDS, min_score)


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


def create_app(
    loader: Callable[[], Any] = preload,
    document_loader: Optional[Callable[[], Any]] = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = build_pool(loader)
        app.state.pool = pool
        app.state.engine = pool
        app.state.document_loader = document_loader or preload_document
        app.state.document_pool = None
        if document_preload_enabled():
            app.state.document_pool = get_or_create_document_pool(
                app, app.state.document_loader, build_pool
            )
        app.state.jobs = JobStore(
            job_root(), runner=lambda files, options: _run_job(app, files, options)
        )
        app.state.ready = True
        logger.info(
            "OCR pool ready: %s session(s), max_side=%s, min_short=%s",
            pool.size,
            effective_max_side(),
            min_short(),
        )
        yield
        app.state.ready = False

    app = FastAPI(
        title=SERVICE_NAME,
        version=SERVICE_VERSION,
        lifespan=lifespan,
        default_response_class=JSONResponse,
    )
    install_guard(app)

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
            "provider": active_provider(),
            "models_ready": ready,
            "workers": worker_count(),
            "max_side": effective_max_side(),
            "min_short": min_short(),
        }

    @app.get("/version", response_model=VersionResponse)
    def version():
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "provider": active_provider(),
            "workers": worker_count(),
            "max_side": effective_max_side(),
            "min_short": min_short(),
        }

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
        preprocess: Optional[str] = Query(None),
        min_score: Optional[float] = Query(None),
        pages: Optional[str] = Query(None),
        max_pages: Optional[int] = Query(None),
        dpi: Optional[float] = Query(None),
        crop: Optional[str] = Query(None),
        profile: Optional[str] = Query(None),
        model: Optional[str] = Query(None),
        fix_digits: Optional[str] = Query(None),
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
            preprocess=preprocess,
            min_score=min_score,
            pages=pages,
            max_pages=max_pages,
            dpi=dpi,
            crop=crop,
            profile=profile,
            model=model,
            fix_digits=fix_digits,
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
        preprocess: Optional[str] = Query(None),
        min_score: Optional[float] = Query(None),
        pages: Optional[str] = Query(None),
        max_pages: Optional[int] = Query(None),
        dpi: Optional[float] = Query(None),
        crop: Optional[str] = Query(None),
        profile: Optional[str] = Query(None),
        model: Optional[str] = Query(None),
        fix_digits: Optional[str] = Query(None),
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
            preprocess=preprocess,
            min_score=min_score,
            pages=pages,
            max_pages=max_pages,
            dpi=dpi,
            crop=crop,
            profile=profile,
            model=model,
            fix_digits=fix_digits,
        )

    @app.post(
        "/ocr/document",
        tags=["document"],
        responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    )
    async def ocr_document(
        file: UploadFile = File(...),
        formats: Optional[list[str]] = Query(None),
        format: Optional[list[str]] = Query(None),
        output: Optional[list[str]] = Query(None),
        formats_form: Optional[list[str]] = Form(None, alias="formats"),
        format_form: Optional[list[str]] = Form(None, alias="format"),
        output_form: Optional[list[str]] = Form(None, alias="output"),
        preprocess: Optional[str] = Query(None),
        min_score: Optional[float] = Query(None),
        pages: Optional[str] = Query(None),
        max_pages: Optional[int] = Query(None),
        dpi: Optional[float] = Query(None),
        crop: Optional[str] = Query(None),
        model: Optional[str] = Query(None),
        fix_digits: Optional[str] = Query(None),
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
            preprocess=preprocess,
            min_score=min_score,
            pages=pages,
            max_pages=max_pages,
            dpi=dpi,
            crop=crop,
            profile=PROFILE_DOCUMENT,
            model=model,
            fix_digits=fix_digits,
        )

    @app.post(
        "/ocr/document/pdf",
        tags=["document"],
        responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    )
    async def ocr_document_pdf(
        file: UploadFile = File(...),
        formats: Optional[list[str]] = Query(None),
        format: Optional[list[str]] = Query(None),
        output: Optional[list[str]] = Query(None),
        formats_form: Optional[list[str]] = Form(None, alias="formats"),
        format_form: Optional[list[str]] = Form(None, alias="format"),
        output_form: Optional[list[str]] = Form(None, alias="output"),
        preprocess: Optional[str] = Query(None),
        min_score: Optional[float] = Query(None),
        pages: Optional[str] = Query(None),
        max_pages: Optional[int] = Query(None),
        dpi: Optional[float] = Query(None),
        crop: Optional[str] = Query(None),
        model: Optional[str] = Query(None),
        fix_digits: Optional[str] = Query(None),
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
            preprocess=preprocess,
            min_score=min_score,
            pages=pages,
            max_pages=max_pages,
            dpi=dpi,
            crop=crop,
            profile=PROFILE_DOCUMENT,
            model=model,
            fix_digits=fix_digits,
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
    async def parse_ktp_document(
        file: UploadFile = File(...),
        preprocess: Optional[str] = Query(None),
        min_score: Optional[float] = Query(None),
        crop: Optional[str] = Query(None),
    ):
        return await _parse_image(
            app, file, parse_ktp, KTP_FIELDS, preprocess, min_score, crop
        )

    @app.post("/detect/artifacts", tags=["artifacts"])
    async def detect_artifact_regions(
        file: UploadFile = File(...),
        preprocess: Optional[str] = Query(None),
    ):
        payload = await _read_upload(file, MAX_IMAGE_BYTES)
        if looks_like_pdf(payload, file.filename, file.content_type):
            raise HTTPException(
                status_code=400,
                detail="Artifact detection accepts an image, not a PDF.",
            )
        try:
            mode = parse_preprocess(preprocess, "light")
        except DocumentError as exc:
            raise _as_http(exc) from exc
        image = _prepare_page(_decode_image(payload), mode, None, False)
        result = await _run_async(app, image)
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "artifacts": detect_artifacts(image, _ocr_lines(result, DEFAULT_MIN_SCORE)),
        }

    @app.post("/jobs", tags=["jobs"])
    async def create_job(
        files: list[UploadFile] = File(...),
        task: str = Form("ocr"),
        formats: Optional[str] = Form(None),
        webhook: Optional[str] = Form(None),
        preprocess: Optional[str] = Form(None),
        min_score: Optional[float] = Form(None),
    ):
        if task not in {"ocr", "ktp", "npwp", "invoice", "document"}:
            raise HTTPException(status_code=400, detail="Unknown task.")
        uploads = []
        for item in files:
            payload = await item.read()
            if payload:
                uploads.append((item.filename or "upload", payload))
        try:
            saved = expand_uploads(uploads)
        except DocumentError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        record = app.state.jobs.create(
            saved,
            {
                "task": task,
                "formats": formats or "json",
                "webhook": webhook,
                "preprocess": preprocess,
                "min_score": min_score,
            },
        )
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "id": record["id"],
            "status": record["status"],
            "task": record["task"],
            "file_count": record["file_count"],
        }

    @app.get("/jobs/{job_id}", tags=["jobs"])
    def job_status(job_id: str):
        record = app.state.jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "id": record["id"],
            "status": record["status"],
            "task": record.get("task"),
            "file_count": record.get("file_count"),
            "error": record.get("error"),
            "webhook_error": record.get("webhook_error"),
        }

    @app.get("/jobs/{job_id}/result", tags=["jobs"])
    def job_result(job_id: str):
        record = app.state.jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if record["status"] != "done":
            raise HTTPException(status_code=409, detail="Job is not finished.")
        result = app.state.jobs.result(job_id)
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "id": job_id,
            "status": "done",
            "result": result,
        }

    @app.post("/parse/npwp", tags=["parsers"])
    async def parse_npwp_document(
        file: UploadFile = File(...),
        preprocess: Optional[str] = Query(None),
        min_score: Optional[float] = Query(None),
        crop: Optional[str] = Query(None),
    ):
        return await _parse_image(
            app, file, parse_npwp, NPWP_FIELDS, preprocess, min_score, crop
        )

    @app.post("/parse/invoice", tags=["parsers"])
    async def parse_invoice_document(
        file: UploadFile = File(...),
        preprocess: Optional[str] = Query(None),
        min_score: Optional[float] = Query(None),
        crop: Optional[str] = Query(None),
    ):
        return await _parse_image(
            app, file, parse_invoice, INVOICE_FIELDS, preprocess, min_score, crop
        )

    return app


def _checked_min_score(value: Optional[float]) -> float:
    if value is None:
        return DEFAULT_MIN_SCORE
    if not 0 <= value <= 1:
        raise HTTPException(
            status_code=400, detail="min_score must be between 0 and 1."
        )
    return float(value)


def _as_http(exc: DocumentError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


_image_loader = LoadImage()


def _decode_image(payload: bytes):
    try:
        return _image_loader(payload)
    except LoadImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _prepare_page(image, mode: str, crop: Optional[str], apply_region: bool):
    try:
        if mode != "off":
            image = preprocess_bgr(image, mode)
        if apply_region and crop:
            image = apply_crop(image, crop)
    except DocumentError as exc:
        raise _as_http(exc) from exc
    return image


async def _parse_image(app, file, parser, field_names, preprocess, min_score, crop):
    payload = await _read_upload(file, MAX_IMAGE_BYTES)
    if looks_like_pdf(payload, file.filename, file.content_type):
        raise HTTPException(
            status_code=400,
            detail="Document parsing accepts an image, not a PDF.",
        )
    try:
        mode = parse_preprocess(preprocess, "full")
    except DocumentError as exc:
        raise _as_http(exc) from exc
    score = _checked_min_score(min_score)
    image = _prepare_page(_decode_image(payload), mode, crop, True)
    result = await _run_async(app, image)
    body = parse_document_body(result, parser, field_names, score)
    body["artifacts"] = detect_artifacts(image, body["lines"])
    return body


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
    preprocess=None,
    min_score=None,
    pages=None,
    max_pages=None,
    dpi=None,
    crop=None,
    profile=None,
    model=None,
    fix_digits=None,
):
    try:
        export_formats = collect_formats(
            formats, format, output, formats_form, format_form, output_form
        )
        resolved_profile = parse_profile(profile)
        # ``model`` and ``fix_digits`` are ignored unless the document profile is on,
        # so a stray query string cannot change the default / KTP path.
        use_fix = parse_fix_digits(fix_digits, False) if resolved_profile == PROFILE_DOCUMENT else False
        if resolved_profile == PROFILE_DOCUMENT:
            parse_document_model(model)
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
    try:
        mode = parse_preprocess(preprocess, "off")
    except DocumentError as exc:
        raise _as_http(exc) from exc
    score = _checked_min_score(min_score)
    if pdf:
        return await asyncio.to_thread(
            _recognize_pdf,
            app,
            payload,
            export_formats,
            mode,
            score,
            pages,
            max_pages,
            dpi,
            crop,
            resolved_profile,
            use_fix,
        )
    if mode == "off" and not crop:
        image = _try_decode(payload)
        result = await _run_async(app, payload if image is None else image, resolved_profile)
    else:
        image = _prepare_page(_decode_image(payload), mode, crop, True)
        result = await _run_async(app, image, resolved_profile)
    artifacts = (
        [] if image is None else detect_artifacts(image, _ocr_lines(result, score))
    )
    body = ocr_body(result, score, artifacts)
    body = _apply_fix_digits_ocr_body(body, use_fix)
    if export_formats == ["json"]:
        return OcrResponse(**body)
    doc = document_body([result], score, [image])
    doc = _apply_fix_digits_document_body(doc, use_fix)
    return render_document(doc, export_formats)


def _recognize_pdf(
    app,
    payload: bytes,
    export_formats: list[str],
    mode: str,
    min_score: float,
    pages,
    max_pages,
    dpi,
    crop,
    profile: str = "default",
    fix_digits: bool = False,
):
    try:
        images = rasterize_pdf(
            payload,
            dpi=_effective_dpi(dpi, profile),
            pages=pages,
            max_pages=max_pages,
        )
    except DocumentError as exc:
        raise _as_http(exc) from exc
    prepared = [
        _prepare_page(image, mode, crop, apply_region=index == 0)
        for index, image in enumerate(images)
    ]
    results = _run_many(app, prepared, profile)
    body = document_body(results, min_score, prepared)
    body = _apply_fix_digits_document_body(body, fix_digits)
    return render_document(body, export_formats)


def _run_job(app, files, options):
    task = options.get("task") or "ocr"
    score = _checked_min_score(options.get("min_score"))
    profile = PROFILE_DOCUMENT if task == "document" else "default"
    default_mode = "off" if task in {"ocr", "document"} else "full"
    try:
        mode = parse_preprocess(options.get("preprocess"), default_mode)
        export_formats = collect_formats(options.get("formats"))
    except DocumentError as exc:
        raise RuntimeError(exc.detail) from exc
    parsers = {
        "ktp": (parse_ktp, KTP_FIELDS),
        "npwp": (parse_npwp, NPWP_FIELDS),
        "invoice": (parse_invoice, INVOICE_FIELDS),
    }
    documents = []
    for name, payload in files:
        if looks_like_pdf(payload, name, None):
            images = rasterize_pdf(
                payload, dpi=_effective_dpi(None, profile) if profile == PROFILE_DOCUMENT else None
            )
        else:
            images = [_decode_image(payload)]
        prepared = [_prepare_page(image, mode, None, False) for image in images]
        results = _run_many(app, prepared, profile)
        if task in {"ocr", "document"}:
            body = document_body(results, score, prepared)
            documents.append(
                {
                    "name": name,
                    "outputs": {
                        export_format: _rendered_output(body, export_format)
                        for export_format in export_formats
                    },
                }
            )
            continue
        parser, field_names = parsers[task]
        documents.append(
            {
                "name": name,
                "pages": [
                    _parsed_page(result, image, parser, field_names, score)
                    for result, image in zip(results, prepared)
                ],
            }
        )
    return {"task": task, "files": documents}


def _parsed_page(result, image, parser, field_names, score: float) -> dict[str, Any]:
    body = parse_document_body(result, parser, field_names, score)
    body["artifacts"] = detect_artifacts(image, body["lines"])
    return body


def _try_decode(payload: bytes):
    try:
        return _image_loader(payload)
    except Exception:
        return None


def _pool(app: FastAPI) -> EnginePool:
    pool = getattr(app.state, "pool", None)
    if pool is None:
        engine = getattr(app.state, "engine", None)
        if engine is None:
            raise HTTPException(status_code=503, detail="OCR engine is not ready.")
        pool = engine if isinstance(engine, EnginePool) else EnginePool([engine])
        app.state.pool = pool
    return pool


def _maybe_downscale(payload, profile: str = "default"):
    if profile == PROFILE_DOCUMENT:
        cap = document_max_side()
        floor = document_min_short()
    else:
        cap = max_side()
        floor = min_short()
    if cap <= 0 or not isinstance(payload, np.ndarray):
        return payload, 1.0, 1.0
    return resize_max_side(payload, cap, min_short=floor)


def _effective_dpi(dpi, profile: str):
    if dpi is not None:
        return dpi
    if profile == PROFILE_DOCUMENT:
        return document_dpi()
    return None


def _apply_fix_digits_ocr_body(body: dict, enabled: bool) -> dict:
    if not enabled:
        return body
    body = dict(body)
    body["lines"] = apply_fix_digits_to_lines(body.get("lines") or [], True)
    return body


def _apply_fix_digits_document_body(body: dict, enabled: bool) -> dict:
    if not enabled:
        return body
    body = dict(body)
    pages = []
    for page in body.get("pages") or []:
        item = dict(page)
        item["lines"] = apply_fix_digits_to_lines(item.get("lines") or [], True)
        item["text"] = apply_fix_digits_to_text(item.get("text") or "", True)
        pages.append(item)
    body["pages"] = pages
    body["text"] = apply_fix_digits_to_text(body.get("text") or "", True)
    return body


def _profile_pool(app: FastAPI, profile: str) -> EnginePool:
    if profile != PROFILE_DOCUMENT:
        return _pool(app)
    loader = getattr(app.state, "document_loader", preload_document)
    return get_or_create_document_pool(app, loader, build_pool)


async def _run_async(app: FastAPI, payload, profile: str = "default"):
    # Keep the event loop free so concurrent HTTP requests can use the pool.
    return await asyncio.to_thread(_run, app, payload, profile)


async def _run_many_async(app: FastAPI, images, profile: str = "default"):
    return await asyncio.to_thread(_run_many, app, images, profile)


def _run(app: FastAPI, payload, profile: str = "default"):
    try:
        image, scale_x, scale_y = _maybe_downscale(payload, profile)
        return scale_result_boxes(_profile_pool(app, profile).run(image), scale_x, scale_y)
    except LoadImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _run_many(app: FastAPI, images, profile: str = "default"):
    try:
        prepared = []
        scales = []
        for item in images:
            image, scale_x, scale_y = _maybe_downscale(item, profile)
            prepared.append(image)
            scales.append((scale_x, scale_y))
        results = _profile_pool(app, profile).run_many(prepared)
        return [
            scale_result_boxes(result, scale_x, scale_y)
            for result, (scale_x, scale_y) in zip(results, scales)
        ]
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
