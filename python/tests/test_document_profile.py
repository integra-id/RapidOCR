# -*- encoding: utf-8 -*-
"""Opt-in document OCR profile. Default / KTP paths must stay unchanged."""

import io
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from PIL import Image

from rapidocr.service.app import _maybe_downscale, create_app
from rapidocr.service.document import (
    PROFILE_DOCUMENT,
    fix_o_in_digit_groups,
    parse_document_model,
    parse_fix_digits,
    parse_profile,
)
from rapidocr.service.pdf import DocumentError
from rapidocr.service.runtime import effective_max_side, min_short, worker_count
from rapidocr.service.version import SERVICE_NAME, SERVICE_VERSION


def _png(width=32, height=32) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class _NamedEngine:
    def __init__(self, name: str, elapse: float = 0.01):
        self.name = name
        self.txts = (name,)
        self.scores = (0.99,)
        self.boxes = np.array(
            [[[0, 0], [10, 0], [10, 1], [0, 1]]],
            dtype=np.float32,
        )
        self.elapse = elapse
        self.calls = 0

    def __call__(self, _payload):
        self.calls += 1
        return self


def _client(default_name="default-engine", document_name="document-engine"):
    default_engine = _NamedEngine(default_name)
    document_engine = _NamedEngine(document_name)
    app = create_app(
        loader=lambda: default_engine,
        document_loader=lambda: document_engine,
    )
    return TestClient(app), default_engine, document_engine


def test_parse_profile_defaults_and_rejects_unknown():
    assert parse_profile(None) == "default"
    assert parse_profile("") == "default"
    assert parse_profile("default") == "default"
    assert parse_profile("KTP") == "default"
    assert parse_profile("document") == PROFILE_DOCUMENT
    assert parse_profile("DOC") == PROFILE_DOCUMENT
    with pytest.raises(DocumentError):
        parse_profile("ktp-v2")
    assert parse_document_model(None) == "tiny"
    assert parse_document_model("small") == "small"
    with pytest.raises(DocumentError):
        parse_document_model("mobile")
    assert parse_fix_digits(None) is False
    assert parse_fix_digits("true") is True


def test_fix_o_in_digit_groups_only_changes_letter_o_in_numbers():
    assert fix_o_in_digit_groups("Rp. 275.0oo.000") == "Rp. 275.000.000"
    assert fix_o_in_digit_groups("Yogi") == "Yogi"


def test_health_and_version_json_stay_the_same():
    with _client()[0] as client:
        health = client.get("/health")
        version = client.get("/version")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "lang": "id",
        "provider": "cpu",
        "models_ready": True,
        "workers": worker_count(),
        "max_side": effective_max_side(),
        "min_short": min_short(),
    }
    assert set(health.json()) == {
        "status",
        "service",
        "version",
        "lang",
        "provider",
        "models_ready",
        "workers",
        "max_side",
        "min_short",
    }
    assert version.json() == {
        "service": "rapidocr-id",
        "version": SERVICE_VERSION,
        "provider": "cpu",
        "workers": worker_count(),
        "max_side": effective_max_side(),
        "min_short": min_short(),
    }


def test_default_ocr_and_parse_ktp_use_default_engine_only():
    client, default_engine, document_engine = _client()
    with client:
        ocr = client.post("/ocr", files={"file": ("page.png", _png(), "image/png")})
        ktp = client.post("/parse/ktp", files={"file": ("ktp.png", _png(), "image/png")})
        npwp = client.post("/parse/npwp", files={"file": ("npwp.png", _png(), "image/png")})
        invoice = client.post(
            "/parse/invoice", files={"file": ("inv.png", _png(), "image/png")}
        )

    assert ocr.status_code == 200
    assert ocr.json()["lines"][0]["text"] == "default-engine"
    assert ktp.status_code == 200
    assert ktp.json()["lines"][0]["text"] == "default-engine"
    assert npwp.status_code == 200
    assert invoice.status_code == 200
    assert default_engine.calls == 4
    assert document_engine.calls == 0
    # Document pool stays lazy when unused.
    assert client.app.state.document_pool is None


def test_document_endpoints_use_document_engine():
    client, default_engine, document_engine = _client()
    with client:
        via_param = client.post(
            "/ocr?profile=document",
            files={"file": ("page.png", _png(), "image/png")},
        )
        via_path = client.post(
            "/ocr/document",
            files={"file": ("page.png", _png(), "image/png")},
        )
        ktp = client.post("/parse/ktp", files={"file": ("ktp.png", _png(), "image/png")})

    assert via_param.status_code == 200
    assert via_param.json()["lines"][0]["text"] == "document-engine"
    assert via_path.status_code == 200
    assert via_path.json()["lines"][0]["text"] == "document-engine"
    assert set(via_path.json()) == set(via_param.json())
    assert ktp.json()["lines"][0]["text"] == "default-engine"
    assert document_engine.calls == 2
    assert default_engine.calls == 1


def test_unknown_profile_is_rejected_and_does_not_run_ocr():
    client, default_engine, document_engine = _client()
    with client:
        response = client.post(
            "/ocr?profile=experimental",
            files={"file": ("page.png", _png(), "image/png")},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "profile must be default or document."}
    assert default_engine.calls == 0
    assert document_engine.calls == 0


def test_document_downscale_is_independent_of_default_cap():
    image = np.zeros((2000, 3000, 3), dtype=np.uint8)
    default_out, _, _ = _maybe_downscale(image, "default")
    document_out, _, _ = _maybe_downscale(image, PROFILE_DOCUMENT)
    # Production default caps the long side at 1280 unless env is changed.
    if effective_max_side() and max(image.shape[:2]) > effective_max_side():
        assert default_out.shape != image.shape
    assert document_out is image
    assert document_out.shape == (2000, 3000, 3)


def test_jobs_default_ocr_task_still_uses_default_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPIDOCR_JOB_DIR", str(tmp_path))
    client, default_engine, document_engine = _client()
    with client:
        created = client.post(
            "/jobs",
            data={"task": "ocr"},
            files=[("files", ("page.png", _png(), "image/png"))],
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        status = "queued"
        for _ in range(50):
            status = client.get(f"/jobs/{job_id}").json()["status"]
            if status in {"done", "error"}:
                break
            time.sleep(0.05)
        result = client.get(f"/jobs/{job_id}/result")
    assert status == "done"

    assert result.status_code == 200
    text = result.json()["result"]["files"][0]["outputs"]["json"]["text"]
    assert "default-engine" in text
    assert document_engine.calls == 0
    assert default_engine.calls >= 1


def test_jobs_document_task_uses_document_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPIDOCR_JOB_DIR", str(tmp_path))
    client, default_engine, document_engine = _client()
    with client:
        created = client.post(
            "/jobs",
            data={"task": "document"},
            files=[("files", ("page.png", _png(), "image/png"))],
        )
        job_id = created.json()["id"]
        status = "queued"
        for _ in range(50):
            status = client.get(f"/jobs/{job_id}").json()["status"]
            if status in {"done", "error"}:
                break
            time.sleep(0.05)
        result = client.get(f"/jobs/{job_id}/result")
    assert status == "done"

    assert result.status_code == 200
    text = result.json()["result"]["files"][0]["outputs"]["json"]["text"]
    assert "document-engine" in text
    assert document_engine.calls >= 1
