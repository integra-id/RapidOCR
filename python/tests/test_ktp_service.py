# -*- encoding: utf-8 -*-
import io
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from rapidocr.postprocess import KTP_FIELDS, parse_ktp
from rapidocr.service.app import create_app, parse_ktp_body
from rapidocr.service.version import SERVICE_NAME, SERVICE_VERSION
from tests.test_ktp_parser import BEKASI_KTP_LINES, GARUT_KTP_LINES, PEMALANG_KTP_LINES

ROOT = Path(__file__).resolve().parents[2]


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


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
    assert response.json() == {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "lang": "id",
        "models_ready": True,
    }


def test_version_endpoint():
    with _client() as client:
        response = client.get("/version")

    assert response.status_code == 200
    body = response.json()
    assert body == {"service": "rapidocr-id", "version": SERVICE_VERSION}
    assert body["version"] == "1.4.0"
    assert response.headers["content-type"].startswith("application/json")


def test_ocr_returns_lines():
    with _client() as client:
        response = client.post(
            "/ocr",
            files={"file": ("ktp.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["service"] == "rapidocr-id"
    assert body["version"] == SERVICE_VERSION
    assert body["lines"][1]["text"] == "3327011112890001"
    assert body["lines"][1]["score"] == 0.98
    assert body["lines"][1]["box"][0] == [0.0, 1.0]
    assert body["elapse"] == 0.01


def test_ktp_parser_is_secondary_endpoint():
    with _client() as client:
        response = client.post(
            "/parse/ktp",
            files={"file": ("ktp.png", _png(), "image/png")},
        )
        alias = client.post(
            "/ktp",
            files={"file": ("ktp.png", _png(), "image/png")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["service"] == "rapidocr-id"
    assert body["version"] == SERVICE_VERSION
    assert body["elapse"] == 0.01
    assert body["min_score"] == 0.5
    assert body["lines"] == [
        {"text": "NIK", "score": 0.99, "below_min_score": False},
        {"text": "3327011112890001", "score": 0.98, "below_min_score": False},
        {"text": "Namá", "score": 0.97, "below_min_score": False},
        {"text": ":RISWANDI", "score": 0.96, "below_min_score": False},
    ]
    assert "box" not in body["lines"][0]
    fields = body["fields"]
    assert tuple(fields) == KTP_FIELDS
    assert fields["nik"] == "3327011112890001"
    assert fields["nama"] == "RISWANDI"
    assert alias.status_code == 200
    assert alias.json()["fields"]["nik"] == fields["nik"]


def _scrambled_boxes(count: int) -> np.ndarray:
    """Y positions that are the reverse of text order."""
    return np.array(
        [
            [
                [0, count - index],
                [1, count - index],
                [1, count - index + 1],
                [0, count - index + 1],
            ]
            for index in range(count)
        ],
        dtype=np.float32,
    )


def test_parse_ktp_uses_text_order_not_boxes():
    samples = (GARUT_KTP_LINES, PEMALANG_KTP_LINES, BEKASI_KTP_LINES)
    for lines in samples:
        result = _Result()
        result.txts = tuple(lines)
        result.scores = tuple(0.9 for _ in lines)
        result.boxes = _scrambled_boxes(len(lines))
        result.elapse = 0.2

        body = parse_ktp_body(result)
        expected = parse_ktp(list(lines))
        scrambled = parse_ktp(result)

        assert body["fields"] == expected
        assert body["fields"] != scrambled
        assert [line["text"] for line in body["lines"]] == list(lines)
        assert all("box" not in line for line in body["lines"])


def test_empty_upload_is_rejected():
    with _client() as client:
        response = client.post(
            "/ktp",
            files={"file": ("ktp.jpg", b"", "image/jpeg")},
        )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Empty image upload."}


def test_production_image_bakes_models_and_publishes_port():
    dockerfile = (ROOT / "docker" / "Dockerfile.rapidocr-id").read_text(
        encoding="utf-8"
    )
    compose = (ROOT / "docker" / "docker-compose.rapidocr-id.yml").read_text(
        encoding="utf-8"
    )
    workflow = (ROOT / ".github" / "workflows" / "publish-rapidocr-id.yml").read_text(
        encoding="utf-8"
    )

    assert "RAPIDOCR_MODEL_DIR=/opt/rapidocr/models" in dockerfile
    assert "python -m rapidocr.service.preload" in dockerfile
    assert "PP-OCRv6_det_small.onnx" in dockerfile
    assert "PP-OCRv6_rec_small.onnx" in dockerfile
    assert "rapidocr-id:" in compose
    assert "8000:8000" in compose
    assert "../python" not in compose
    assert "/data/jobs" in compose
    assert "ghcr.io/integra-id/rapidocr-id" in workflow
    assert 'tags:\n      - "v*"' in workflow or '- "v*"' in workflow
