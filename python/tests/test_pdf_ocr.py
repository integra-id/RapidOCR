# -*- encoding: utf-8 -*-
import io

import numpy as np
import pypdfium2 as pdfium
from fastapi.testclient import TestClient

from rapidocr.service.app import create_app
from rapidocr.service.pdf import MAX_PDF_PAGES, rasterize_pdf


def blank_pdf(page_count: int) -> bytes:
    document = pdfium.PdfDocument.new()
    pages = [document.new_page(72, 72) for _ in range(page_count)]
    buffer = io.BytesIO()
    document.save(buffer)
    for page in pages:
        page.close()
    document.close()
    return buffer.getvalue()


class _PageResult:
    def __init__(self, label: str):
        self.txts = (label,)
        self.scores = (0.8,)
        self.boxes = np.array([[[1, 1], [20, 1], [20, 10], [1, 10]]], dtype=np.float32)
        self.elapse = 0.25


class _Engine:
    def __init__(self):
        self.calls = 0

    def __call__(self, _image):
        self.calls += 1
        return _PageResult(f"LINE-{self.calls}")


def _client():
    return TestClient(create_app(loader=_Engine))


def test_rasterize_two_page_pdf():
    images = rasterize_pdf(blank_pdf(2))

    assert len(images) == 2
    assert images[0].ndim == 3 and images[0].shape[2] == 3
    assert images[0].shape[0] == 144


def test_pdf_page_cap():
    try:
        rasterize_pdf(blank_pdf(MAX_PDF_PAGES + 1))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 413
        return
    raise AssertionError("oversized PDF was accepted")


def _post(path: str, payload: bytes, **kwargs):
    with _client() as client:
        return client.post(
            path,
            files={"file": ("notes.pdf", payload, "application/pdf")},
            **kwargs,
        )


def test_pdf_formats():
    payload = blank_pdf(2)
    json_response = _post("/ocr/pdf?format=json", payload)
    text_response = _post("/ocr/pdf", payload, data={"output": "txt"})
    markdown_response = _post("/ocr?format=md", payload)
    html_response = _post("/ocr/pdf?output=html", payload)

    assert json_response.headers["content-type"].startswith("application/json")
    body = json_response.json()
    assert body["service"] == "rapidocr-id"
    assert body["version"] == "1.5.0"
    assert body["page_count"] == 2
    assert body["elapse"] == 0.5
    assert body["pages"][0]["page"] == 1
    assert body["pages"][0]["lines"][0]["text"] == "LINE-1"
    assert body["pages"][1]["lines"][0]["text"] == "LINE-2"
    assert "LINE-1" in body["text"] and "LINE-2" in body["text"]
    assert body["pages"][0]["lines"][0]["box"][0] == [1.0, 1.0]

    assert text_response.headers["content-type"].startswith("text/plain")
    assert "--- page 1 ---" in text_response.text
    assert "LINE-2" in text_response.text

    assert markdown_response.headers["content-type"].startswith("text/markdown")
    assert "# Page 1" in markdown_response.text
    assert "LINE-1" in markdown_response.text

    assert html_response.headers["content-type"].startswith("text/html")
    assert "<h1>Page 2</h1>" in html_response.text
    assert "<p>LINE-2</p>" in html_response.text


def test_multi_format_uses_one_ocr_pass():
    engine = _Engine()
    payload = blank_pdf(2)
    app = create_app(loader=lambda: engine)
    with TestClient(app) as client:
        combined = client.post(
            "/ocr/pdf?formats=json,md,html",
            files={"file": ("notes.pdf", payload, "application/pdf")},
        )
        repeated = client.post(
            "/ocr?format=txt&format=json",
            files={"file": ("notes.pdf", payload, "application/pdf")},
        )

    assert engine.calls == 4
    assert combined.headers["content-type"].startswith("application/json")
    body = combined.json()
    assert body["service"] == "rapidocr-id"
    assert body["version"] == "1.5.0"
    assert body["page_count"] == 2
    assert body["elapse"] == 0.5
    assert set(body["outputs"]) == {"json", "markdown", "html"}
    assert body["outputs"]["json"]["pages"][0]["lines"][0]["text"] == "LINE-1"
    assert body["outputs"]["json"]["pages"][1]["lines"][0]["text"] == "LINE-2"
    assert "# Page 1" in body["outputs"]["markdown"]
    assert "<p>LINE-2</p>" in body["outputs"]["html"]

    again = repeated.json()
    assert set(again["outputs"]) == {"txt", "json"}
    assert "--- page 1 ---" in again["outputs"]["txt"]
    assert again["outputs"]["json"]["pages"][0]["lines"][0]["text"] == "LINE-3"


def test_unknown_format_and_ktp_reject_pdf():
    payload = blank_pdf(1)
    with _client() as client:
        bad_format = client.post(
            "/ocr/pdf?format=docx",
            files={"file": ("notes.pdf", payload, "application/pdf")},
        )
        ktp = client.post(
            "/parse/ktp",
            files={"file": ("notes.pdf", payload, "application/pdf")},
        )
        not_pdf = client.post(
            "/ocr/pdf",
            files={"file": ("photo.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        )

    assert bad_format.status_code == 400
    assert bad_format.headers["content-type"].startswith("application/json")
    assert ktp.status_code == 400
    assert "PDF" in ktp.json()["detail"]
    assert not_pdf.status_code == 400
