# -*- encoding: utf-8 -*-
"""Rasterize PDF pages for the rapidocr-id service.

pypdfium2 ships its own pdfium build, so the image does not need poppler.
"""

from __future__ import annotations

import html
from typing import Any, Optional

import cv2
import numpy as np

from rapidocr.service.layout import detect_tables, layout_html, layout_markdown, render_csv

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
    "csv": "csv",
}

MEDIA_TYPES = {
    "json": "application/json",
    "txt": "text/plain; charset=utf-8",
    "markdown": "text/markdown; charset=utf-8",
    "html": "text/html; charset=utf-8",
    "csv": "text/csv; charset=utf-8",
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


def parse_page_selection(
    count: int, pages: Optional[str], max_pages: Optional[int]
) -> list[int]:
    """Return zero-based page indexes to rasterize.

    ``pages`` is 1-based, for example ``1-3`` or ``1,3,5``. Without it, every
    page is used. A file with more than 20 pages is rejected unless ``pages``
    or ``max_pages`` narrows the work to at most 20 pages.
    """
    if max_pages is not None and not 1 <= max_pages <= MAX_PDF_PAGES:
        raise DocumentError(f"max_pages must be between 1 and {MAX_PDF_PAGES}.")
    limit = MAX_PDF_PAGES if max_pages is None else max_pages

    if pages:
        indexes = _parse_page_spec(pages, count)
    else:
        indexes = list(range(count))

    if pages is None and max_pages is None and count > MAX_PDF_PAGES:
        raise DocumentError(
            f"PDF has {count} pages; the limit is {MAX_PDF_PAGES}.",
            status_code=413,
        )
    if len(indexes) > limit:
        if pages is None:
            return indexes[:limit]
        raise DocumentError(
            f"Selected {len(indexes)} pages; the limit is {limit}.",
            status_code=413,
        )
    return indexes


def render_scale(dpi: Optional[float]) -> float:
    if dpi is None:
        return PDF_RENDER_SCALE
    if dpi < 72 or dpi > 300:
        raise DocumentError("dpi must be between 72 and 300.")
    return float(dpi) / 72.0


def _parse_page_spec(spec: str, count: int) -> list[int]:
    indexes: list[int] = []
    for part in spec.split(","):
        piece = part.strip()
        if not piece:
            continue
        try:
            if "-" in piece:
                start_text, end_text = piece.split("-", 1)
                start, end = int(start_text), int(end_text)
                if start > end:
                    raise ValueError
                numbers = range(start, end + 1)
            else:
                numbers = [int(piece)]
        except ValueError as exc:
            raise DocumentError("pages must look like 1-3 or 1,3,5.") from exc
        for number in numbers:
            if number < 1 or number > count:
                raise DocumentError(f"Page {number} is outside 1..{count}.")
            zero = number - 1
            if zero not in indexes:
                indexes.append(zero)
    if not indexes:
        raise DocumentError("pages must look like 1-3 or 1,3,5.")
    return indexes


def rasterize_pdf(
    payload: bytes,
    dpi: Optional[float] = None,
    pages: Optional[str] = None,
    max_pages: Optional[int] = None,
) -> list[np.ndarray]:
    """Return one BGR image per selected page, in the requested order."""
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(payload)
    except Exception as exc:
        raise DocumentError("The upload is not a readable PDF.") from exc

    try:
        count = len(document)
        if count < 1:
            raise DocumentError("The PDF has no pages.")
        indexes = parse_page_selection(count, pages, max_pages)
        scale = render_scale(dpi)

        images: list[np.ndarray] = []
        for index in indexes:
            page = document[index]
            bitmap = page.render(scale=scale)
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


def page_markdown(lines: list[dict[str, Any]]) -> str:
    if not lines:
        return ""
    rendered = layout_markdown(lines)
    if rendered:
        return rendered
    return "\n".join(line["text"] for line in lines if line.get("text"))


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
        body = page.get("html_body")
        if not body:
            body = "".join(
                f"<p>{html.escape(line['text'])}</p>"
                for line in page.get("lines") or []
                if line.get("text")
            )
        sections.append(f"<section><h1>Page {page['page']}</h1>{body}</section>")
    body = "".join(sections)
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>rapidocr-id</title></head><body>"
        f"{body}</body></html>\n"
    )
