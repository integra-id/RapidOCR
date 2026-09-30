# -*- encoding: utf-8 -*-
"""Rasterize PDF pages for the rapidocr-id service.

pypdfium2 ships its own pdfium build, so the image does not need poppler.
"""

from __future__ import annotations

import html
from typing import Any, Optional

import cv2
import numpy as np

from rapidocr.utils.to_markdown import ToMarkdown

MAX_PDF_PAGES = 20
PDF_RENDER_SCALE = 2.0


class DocumentError(Exception):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


_FORMAT_ALIASES = {
    "json": "json",
    "txt": "txt",
    "text": "txt",
    "plain": "txt",
    "md": "markdown",
    "markdown": "markdown",
    "html": "html",
}

MEDIA_TYPES = {
    "json": "application/json",
    "txt": "text/plain; charset=utf-8",
    "markdown": "text/markdown; charset=utf-8",
    "html": "text/html; charset=utf-8",
}


def looks_like_pdf(
    payload: bytes, filename: Optional[str] = None, content_type: Optional[str] = None
) -> bool:
    if payload.startswith(b"%PDF"):
        return True
    name = (filename or "").lower()
    media = (content_type or "").split(";", 1)[0].strip().lower()
    return name.endswith(".pdf") or media == "application/pdf"


def normalize_format(raw: str) -> str:
    key = str(raw).strip().lower()
    if key not in _FORMAT_ALIASES:
        allowed = "json, txt, md, html"
        raise DocumentError(f"Unknown format {raw!r}. Use {allowed}.")
    return _FORMAT_ALIASES[key]


def collect_formats(*candidates: Any) -> list[str]:
    """Read comma-separated ``formats`` and repeated ``format`` / ``output``."""
    names: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        items = candidate if isinstance(candidate, (list, tuple)) else [candidate]
        for item in items:
            if item is None:
                continue
            for piece in str(item).split(","):
                piece = piece.strip()
                if piece:
                    names.append(normalize_format(piece))
    if not names:
        return ["json"]
    unique: list[str] = []
    for name in names:
        if name not in unique:
            unique.append(name)
    return unique


def rasterize_pdf(payload: bytes) -> list[np.ndarray]:
    """Return one BGR image per page, in PDF order."""
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(payload)
    except Exception as exc:
        raise DocumentError("The upload is not a readable PDF.") from exc

    try:
        count = len(document)
        if count < 1:
            raise DocumentError("The PDF has no pages.")
        if count > MAX_PDF_PAGES:
            raise DocumentError(
                f"PDF has {count} pages; the limit is {MAX_PDF_PAGES}.",
                status_code=413,
            )

        images: list[np.ndarray] = []
        for index in range(count):
            page = document[index]
            bitmap = page.render(scale=PDF_RENDER_SCALE)
            try:
                rgb = bitmap.to_numpy()
                images.append(_to_bgr(rgb))
            finally:
                bitmap.close()
                page.close()
        return images
    finally:
        document.close()


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def page_plain_text(lines: list[dict[str, Any]]) -> str:
    return "\n".join(line["text"] for line in lines if line.get("text"))


def page_markdown(result) -> str:
    texts = getattr(result, "txts", None) or ()
    if not texts:
        return ""
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) == len(texts):
        rendered = ToMarkdown.to(np.asarray(boxes), tuple(texts))
        if rendered and not rendered.startswith("没有检测到"):
            return rendered
    return "\n".join(str(text) for text in texts if text)


def render_text(pages: list[dict[str, Any]]) -> str:
    chunks = []
    for page in pages:
        chunks.append(f"--- page {page['page']} ---")
        chunks.append(page.get("text") or "")
    return "\n".join(chunks).rstrip() + "\n"


def render_markdown(pages: list[dict[str, Any]]) -> str:
    chunks = []
    for page in pages:
        body = page.get("markdown") or page.get("text") or ""
        chunks.append(f"# Page {page['page']}\n\n{body}".rstrip())
    return "\n\n".join(chunks) + "\n"


def render_html(pages: list[dict[str, Any]]) -> str:
    sections = []
    for page in pages:
        paragraphs = "".join(
            f"<p>{html.escape(line['text'])}</p>"
            for line in page.get("lines") or []
            if line.get("text")
        )
        sections.append(f"<section><h1>Page {page['page']}</h1>{paragraphs}</section>")
    body = "".join(sections)
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>rapidocr-id</title></head><body>"
        f"{body}</body></html>\n"
    )
