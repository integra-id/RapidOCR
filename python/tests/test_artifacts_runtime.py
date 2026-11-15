# -*- encoding: utf-8 -*-
import pytest

pytest.importorskip("fastapi")

import cv2
import numpy as np

from rapidocr.service.artifacts import detect_artifacts
from rapidocr.service.runtime import (
    active_provider,
    default_worker_count,
    effective_max_side,
    engine_params,
    intra_op_threads,
    max_side,
    requested_provider,
    worker_count,
)


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
    params = engine_params()
    assert params["EngineConfig.onnxruntime.use_cuda"] is True
    assert params["Global.max_side_len"] == effective_max_side()
    assert active_provider() == "cpu"


def test_default_provider_is_cpu(monkeypatch):
    monkeypatch.delenv("RAPIDOCR_PROVIDER", raising=False)

    assert requested_provider() == "cpu"
    params = engine_params()
    assert "EngineConfig.onnxruntime.use_cuda" not in params
    assert params["Global.max_side_len"] == effective_max_side()
    assert params["EngineConfig.onnxruntime.intra_op_num_threads"] == intra_op_threads()
    assert params["EngineConfig.onnxruntime.inter_op_num_threads"] == 1
    assert active_provider() == "cpu"


def test_default_worker_count_scales_with_cpu():
    assert default_worker_count(1) == 1
    assert default_worker_count(2) == 2
    assert default_worker_count(4) == 2
    assert default_worker_count(6) == 3
    assert default_worker_count(8) == 4
    assert default_worker_count(16) == 4


def test_worker_count_reads_env(monkeypatch):
    monkeypatch.setenv("RAPIDOCR_WORKERS", "3")
    assert worker_count() == 3
    monkeypatch.setenv("RAPIDOCR_WORKERS", "0")
    assert worker_count() == 1
    monkeypatch.setenv("RAPIDOCR_WORKERS", "99")
    assert worker_count() == 16
    monkeypatch.setenv("RAPIDOCR_WORKERS", "nope")
    assert worker_count() == default_worker_count()
    monkeypatch.delenv("RAPIDOCR_WORKERS", raising=False)
    assert worker_count() == default_worker_count()


def test_max_side_reads_env(monkeypatch):
    monkeypatch.delenv("RAPIDOCR_MAX_SIDE", raising=False)
    assert max_side() == 1280
    assert effective_max_side() == 1280
    monkeypatch.setenv("RAPIDOCR_MAX_SIDE", "1600")
    assert max_side() == 1600
    assert effective_max_side() == 1600
    monkeypatch.setenv("RAPIDOCR_MAX_SIDE", "0")
    assert max_side() == 0
    assert effective_max_side() == 2000
    monkeypatch.setenv("RAPIDOCR_MAX_SIDE", "8")
    assert max_side() == 32
    monkeypatch.setenv("RAPIDOCR_MAX_SIDE", "bad")
    assert max_side() == 1280


def test_intra_op_threads_split_cores(monkeypatch):
    monkeypatch.setenv("RAPIDOCR_WORKERS", "4")
    monkeypatch.delenv("RAPIDOCR_INTRA_THREADS", raising=False)
    assert intra_op_threads(workers=4, cores=8) == 2
    monkeypatch.setenv("RAPIDOCR_INTRA_THREADS", "3")
    assert intra_op_threads(workers=4, cores=8) == 3

