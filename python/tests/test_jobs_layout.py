# -*- encoding: utf-8 -*-
import io
import time
import zipfile

from fastapi.testclient import TestClient
from PIL import Image

from rapidocr.service.auth import reset_rate_limits
from rapidocr.service.app import create_app
from rapidocr.service.layout import (
    detect_tables,
    layout_html,
    layout_markdown,
    tables_to_csv,
)
from tests.test_pdf_ocr import _Engine


def _png(name: str = "page.png") -> tuple[str, bytes]:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buffer, format="PNG")
    return name, buffer.getvalue()


def _table_lines():
    def box(x, y, w=40, h=12):
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    return [
        {"text": "Nama", "box": box(0, 0)},
        {"text": "Budi", "box": box(80, 0)},
        {"text": "Kota", "box": box(0, 24)},
        {"text": "Garut", "box": box(80, 24)},
        {"text": "- Catatan", "box": box(0, 80, w=120, h=12)},
    ]


def test_table_csv_and_layout():
    lines = _table_lines()
    tables = detect_tables(lines)
    assert tables == [[["Nama", "Budi"], ["Kota", "Garut"]]]
    csv_text = tables_to_csv(tables)
    assert "Nama,Budi" in csv_text
    assert "Kota,Garut" in csv_text
    markdown = layout_markdown(lines)
    assert "| Nama | Budi |" in markdown
    assert "- Catatan" in markdown
    html_text = layout_html(lines)
    assert "<table>" in html_text
    assert "<li>- Catatan</li>" in html_text or "<li>" in html_text


def test_api_key_protects_routes_except_health(monkeypatch):
    monkeypatch.setenv("RAPIDOCR_API_KEY", "secret-key")
    monkeypatch.setenv("RAPIDOCR_RATE_LIMIT", "0")
    with TestClient(create_app(loader=_Engine)) as client:
        open_health = client.get("/health")
        blocked = client.get("/version")
        allowed = client.get("/version", headers={"X-API-Key": "secret-key"})
        bearer = client.get("/version", headers={"Authorization": "Bearer secret-key"})

    assert open_health.status_code == 200
    assert blocked.status_code == 401
    assert allowed.status_code == 200
    assert bearer.status_code == 200


def test_rate_limit(monkeypatch):
    monkeypatch.delenv("RAPIDOCR_API_KEY", raising=False)
    monkeypatch.setenv("RAPIDOCR_RATE_LIMIT", "2")
    reset_rate_limits()
    with TestClient(create_app(loader=_Engine)) as client:
        assert client.get("/version").status_code == 200
        assert client.get("/version").status_code == 200
        denied = client.get("/version")
    assert denied.status_code == 429


def test_job_create_status_and_result(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPIDOCR_JOB_DIR", str(tmp_path))
    monkeypatch.setenv("RAPIDOCR_RATE_LIMIT", "0")
    monkeypatch.delenv("RAPIDOCR_API_KEY", raising=False)
    engine = _Engine()
    first = _png("satu.png")
    second = _png("dua.png")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(first[0], first[1])
        bundle.writestr(second[0], second[1])

    with TestClient(create_app(loader=lambda: engine)) as client:
        created = client.post(
            "/jobs",
            files={"files": ("batch.zip", archive.getvalue(), "application/zip")},
            data={"task": "ocr", "formats": "json"},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        assert created.json()["file_count"] == 2
        status = "queued"
        for _ in range(50):
            status = client.get(f"/jobs/{job_id}").json()["status"]
            if status in {"done", "error"}:
                break
            time.sleep(0.05)
        result = client.get(f"/jobs/{job_id}/result")

    assert status == "done"
    assert result.status_code == 200
    names = [item["name"] for item in result.json()["result"]["files"]]
    assert names == ["satu.png", "dua.png"]
    assert engine.calls == 2
