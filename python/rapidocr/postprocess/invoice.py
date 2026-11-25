# -*- encoding: utf-8 -*-
"""Parse an Indonesian invoice or faktur pajak from OCR text lines."""

from __future__ import annotations

import re
from typing import Optional, Sequence

from .fuzzy_labels import compact, fold, split_label

INVOICE_FIELDS = (
    "nomor_invoice",
    "tanggal",
    "nama_penjual",
    "nama_pembeli",
    "npwp_penjual",
    "npwp_pembeli",
    "subtotal",
    "ppn",
    "total",
    "currency",
)

_LABELS = {
    "nomor_invoice": (
        "nomor faktur",
        "no faktur",
        "faktur pajak",
        "invoice no",
        "no invoice",
        "nomor invoice",
        "invoice number",
    ),
    "tanggal": ("tanggal", "tanggal faktur", "invoice date"),
    "nama_penjual": ("nama penjual", "penjual", "pkp penjual"),
    "nama_pembeli": ("nama pembeli", "pembeli", "nama lawan transaksi"),
    "npwp_penjual": ("npwp penjual",),
    "npwp_pembeli": ("npwp pembeli",),
    "subtotal": ("dpp", "dasar pengenaan pajak", "subtotal", "sub total"),
    "ppn": ("ppn", "pajak pertambahan nilai"),
    "total": ("total", "jumlah", "grand total", "total bayar"),
}


def parse_invoice(lines: Sequence[str]) -> dict[str, Optional[str]]:
    fields: dict[str, Optional[str]] = {key: None for key in INVOICE_FIELDS}
    items = [str(line).strip() for line in lines if line and str(line).strip()]
    index = 0
    while index < len(items):
        field, value = split_label(items[index], _LABELS)
        if field is None:
            index += 1
            continue
        if (
            not value
            and index + 1 < len(items)
            and split_label(items[index + 1], _LABELS)[0] is None
        ):
            value = items[index + 1]
            index += 2
        else:
            index += 1
        cleaned = _clean(field, value)
        if cleaned and fields[field] is None:
            fields[field] = cleaned
    if fields["currency"] is None:
        fields["currency"] = _currency(items)
    return fields


def _clean(field: str, value: str) -> Optional[str]:
    text = re.sub(r"^[\s:：]+", "", value or "").strip()
    if not text:
        return None
    if field in {"npwp_penjual", "npwp_pembeli"}:
        return _npwp_digits(text)
    if field in {"subtotal", "ppn", "total"}:
        return _money(text)
    if field == "tanggal":
        match = re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", text)
        return match.group() if match else re.sub(r"\s+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" ,")


def _npwp_digits(text: str) -> Optional[str]:
    match = re.search(r"(?:\d[\d.\-\s]{8,}\d)", text or "")
    if match is None:
        return None
    digits = re.sub(r"\D", "", match.group())
    if len(digits) in {15, 16}:
        return digits
    return None


def _money(text: str) -> Optional[str]:
    match = re.search(r"\d[\d.]*(?:,\d+)?", text.replace(" ", ""))
    if match is None:
        return None
    token = match.group()
    if "," in token:
        whole, fraction = token.split(",", 1)
        whole = whole.replace(".", "")
        return f"{whole}.{fraction}"
    if token.count(".") > 1 or re.fullmatch(r"\d{1,3}(\.\d{3})+", token):
        return token.replace(".", "")
    return token


def _currency(lines: Sequence[str]) -> Optional[str]:
    blob = " ".join(fold(line) for line in lines)
    compact_blob = compact(blob)
    if "idr" in compact_blob or re.search(r"\brp\b", blob):
        return "IDR"
    if "usd" in compact_blob or "$" in blob:
        return "USD"
    return None
