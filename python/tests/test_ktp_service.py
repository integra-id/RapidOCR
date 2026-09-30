# -*- encoding: utf-8 -*-
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from rapidocr.postprocess import KTP_FIELDS
from rapidocr.service.app import create_app

ROOT = Path(__file__).resolve().parents[2]


class _Result:
    def __init__(self):
        self.txts = (
            "NIK",
            "3327011112890001",
            "Namá",
            ":RISWANDI",
        )
        self.scores = (0.99, 0.98, 0.97, 0.96)
        self.boxes = np.array(
            [[[0, i], [10, i], [10, i + 1], [0, i + 1]] for i in range(4)],
            dtype=np.float32,
        )
        self.elapse = 0.01

    def __call__(self, _payload):
        return self


def _client():
    engine = _Result()
    app = create_app(loader=lambda: engine)
    return TestClient(app)


def test_health_reports_indonesian_models_ready():
    with _client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "lang": "id", "models_ready": True}


def test_ocr_returns_lines():
    with _client() as client:
        response = client.post(
            "/ocr",
            files={"file": ("ktp.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["lines"][1]["text"] == "3327011112890001"
    assert body["lines"][1]["box"][0] == [0.0, 1.0]
    assert body["elapse"] == 0.01


def test_ktp_returns_structured_fields():
    with _client() as client:
        response = client.post(
            "/ktp",
            files={"file": ("ktp.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        )

    assert response.status_code == 200
    fields = response.json()["fields"]
    assert tuple(fields) == KTP_FIELDS
    assert fields["nik"] == "3327011112890001"
    assert fields["nama"] == "RISWANDI"


def test_empty_upload_is_rejected():
    with _client() as client:
        response = client.post(
            "/ktp",
            files={"file": ("ktp.jpg", b"", "image/jpeg")},
        )

    assert response.status_code == 400


def test_production_image_bakes_models_and_publishes_port():
    dockerfile = (ROOT / "docker" / "Dockerfile.ktp").read_text(encoding="utf-8")
    compose = (ROOT / "docker" / "docker-compose.ktp.yml").read_text(encoding="utf-8")

    assert "RAPIDOCR_MODEL_DIR=/opt/rapidocr/models" in dockerfile
    assert "python -m rapidocr.service.preload" in dockerfile
    assert "8000:8000" in compose
    assert "volumes:" not in compose
