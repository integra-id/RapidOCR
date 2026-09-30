# -*- encoding: utf-8 -*-
import cv2
import numpy as np

from rapidocr.service.artifacts import detect_artifacts
from rapidocr.service.runtime import active_provider, engine_params, requested_provider


def test_red_disc_is_a_seal():
    image = np.full((360, 360, 3), 255, dtype=np.uint8)
    cv2.circle(image, (180, 180), 55, (30, 30, 220), -1)

    artifacts = detect_artifacts(image)

    assert any(item["type"] == "seal" and item["score"] >= 0.35 for item in artifacts)


def test_wide_stroke_is_a_signature():
    image = np.full((400, 640, 3), 255, dtype=np.uint8)
    points = np.array(
        [[x, 300 + int(18 * np.sin(x / 14.0))] for x in range(40, 560, 6)],
        dtype=np.int32,
    )
    cv2.polylines(image, [points], False, (25, 25, 25), 2)

    artifacts = detect_artifacts(image)

    assert any(item["type"] == "signature" for item in artifacts)


def test_ocr_phrase_adds_signature_cue():
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    lines = [
        {
            "text": "Tanda tangan",
            "score": 0.9,
            "box": [[20, 140], [180, 140], [180, 170], [20, 170]],
        }
    ]

    artifacts = detect_artifacts(image, lines)

    assert any(item["type"] == "signature" for item in artifacts)


def test_cuda_request_without_provider_uses_cpu(monkeypatch):
    monkeypatch.setenv("RAPIDOCR_PROVIDER", "cuda")

    assert requested_provider() == "cuda"
    assert engine_params()["EngineConfig.onnxruntime.use_cuda"] is True
    assert active_provider() == "cpu"


def test_default_provider_is_cpu(monkeypatch):
    monkeypatch.delenv("RAPIDOCR_PROVIDER", raising=False)

    assert requested_provider() == "cpu"
    assert engine_params() == {}
    assert active_provider() == "cpu"
