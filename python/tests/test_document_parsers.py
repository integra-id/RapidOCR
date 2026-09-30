# -*- encoding: utf-8 -*-
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rapidocr.postprocess.invoice import parse_invoice
from rapidocr.postprocess.npwp import parse_npwp
from rapidocr.service.app import create_app
from rapidocr.service.pdf import parse_page_selection, rasterize_pdf
from rapidocr.service.preprocess import apply_crop, preprocess_bgr
from tests.test_pdf_ocr import _Engine, blank_pdf

NPWP_LINES = [
    "KEMENTERIAN KEUANGAN",
    "NPWP",
    "01.234.567.8-901.000",
    "Nama",
    ":BUDI SANTOSO",
    "Alamat",
    "JL. MERDEKA NO 1",
    "Jenis WP",
    "ORANG PRIBADI",
]

INVOICE_LINES = [
    "FAKTUR PAJAK",
    "Nomor Faktur: 010.000-24.00000001",
    "Tanggal: 15-03-2024",
    "Nama Penjual: PT MAJU JAYA",
    "NPWP Penjual: 01.234.567.8-901.000",
    "Nama Pembeli: PT PEMBELI SEJAHTERA",
    "NPWP Pembeli: 02.345.678.9-012.000",
    "DPP: 1.000.000",
    "PPN: 110.000",
    "Total: Rp 1.110.000",
]


def test_parse_npwp_lines():
    fields = parse_npwp(NPWP_LINES)

    assert fields["npwp"] == "012345678901000"
    assert fields["nama"] == "BUDI SANTOSO"
    assert fields["alamat"] == "JL. MERDEKA NO 1"
    assert fields["jenis_wp"] == "ORANG PRIBADI"


def test_parse_invoice_lines():
    fields = parse_invoice(INVOICE_LINES)

    assert fields["nomor_invoice"] == "010.000-24.00000001"
    assert fields["tanggal"] == "15-03-2024"
    assert fields["nama_penjual"] == "PT MAJU JAYA"
    assert fields["nama_pembeli"] == "PT PEMBELI SEJAHTERA"
    assert fields["npwp_penjual"] == "012345678901000"
    assert fields["npwp_pembeli"] == "023456789012000"
    assert fields["subtotal"] == "1000000"
    assert fields["ppn"] == "110000"
    assert fields["total"] == "1110000"
    assert fields["currency"] == "IDR"


def test_missing_parser_fields_are_null():
    assert parse_npwp(["halo"])["nama"] is None
    assert parse_invoice(["catatan kosong"])["total"] is None


def test_page_selection_and_dpi_cap():
    assert parse_page_selection(5, "1-3", None) == [0, 1, 2]
    assert parse_page_selection(5, "1,3,5", None) == [0, 2, 4]
    assert parse_page_selection(30, None, 2) == [0, 1]
    with pytest.raises(Exception) as exc:
        parse_page_selection(30, None, None)
    assert exc.value.status_code == 413

    images = rasterize_pdf(blank_pdf(3), dpi=72, pages="1,3")
    assert len(images) == 2
    assert images[0].shape[0] == 72


def test_deskew_straightens_a_tilted_bar():
    canvas = np.full((240, 360, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (40, 100), (320, 130), (0, 0, 0), -1)
    matrix = cv2.getRotationMatrix2D((180, 120), 8, 1.0)
    tilted = cv2.warpAffine(canvas, matrix, (360, 240), borderValue=(255, 255, 255))
    from rapidocr.service.preprocess import _skew_degrees

    fixed = preprocess_bgr(tilted, "full")

    assert abs(_skew_degrees(fixed)) < abs(_skew_degrees(tilted))


def test_crop_and_clean_page_skips_heavy_rotation():
    image = np.full((40, 50, 3), 255, dtype=np.uint8)
    cropped = apply_crop(image, "5,6,10,8")
    assert cropped.shape[:2] == (8, 10)

    clean = np.full((120, 160, 3), 240, dtype=np.uint8)
    assert preprocess_bgr(clean, "full").shape == clean.shape


def test_min_score_nulls_uncertain_parser_fields():
    class _LowName:
        txts = ("NPWP", "01.234.567.8-901.000", "Nama", "BUDI")
        scores = (0.99, 0.95, 0.9, 0.2)
        boxes = None
        elapse = 0.1

        def __call__(self, _image):
            return self

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    app = create_app(loader=_LowName)
    with TestClient(app) as client:
        response = client.post(
            "/parse/npwp?min_score=0.5&preprocess=false",
            files={"file": ("npwp.png", buffer.getvalue(), "image/png")},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["fields"]["npwp"] == "012345678901000"
    assert body["fields"]["nama"] is None
    assert body["lines"][3]["below_min_score"] is True
    assert body["field_scores"]["npwp"] == 0.95


def test_selected_pdf_pages_are_one_pass():
    engine = _Engine()
    app = create_app(loader=lambda: engine)
    with TestClient(app) as client:
        response = client.post(
            "/ocr/pdf?pages=1,3&formats=txt,json",
            files={"file": ("notes.pdf", blank_pdf(3), "application/pdf")},
        )

    assert engine.calls == 2
    body = response.json()
    assert body["page_count"] == 2
    assert set(body["outputs"]) == {"txt", "json"}
