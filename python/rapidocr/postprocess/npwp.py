# -*- encoding: utf-8 -*-
"""Parse an Indonesian NPWP card from OCR text lines."""

from __future__ import annotations

import re
from typing import Optional, Sequence

from .fuzzy_labels import split_label

NPWP_FIELDS = ("npwp", "nama", "alamat", "jenis_wp")

_LABELS = {
    "npwp": ("npwp", "nomor pokok wajib pajak"),
    "nama": ("nama", "nama wp", "nama wajib pajak"),
    "alamat": ("alamat",),
    "jenis_wp": ("jenis wp", "jenis wajib pajak"),
}


def parse_npwp(lines: Sequence[str]) -> dict[str, Optional[str]]:
    fields: dict[str, Optional[str]] = {key: None for key in NPWP_FIELDS}
    index = 0
    items = [str(line).strip() for line in lines if line and str(line).strip()]
    while index < len(items):
        field, value = split_label(items[index], _LABELS)
        if field is None:
            if fields["npwp"] is None:
                digits = _npwp_digits(items[index])
                if digits:
                    fields["npwp"] = digits
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
        if cleaned:
            fields[field] = cleaned
    return fields


def _clean(field: str, value: str) -> Optional[str]:
    text = re.sub(r"^[\s:：]+", "", value or "").strip()
    if not text:
        return None
    if field == "npwp":
        return _npwp_digits(text)
    if field == "jenis_wp":
        return re.sub(r"\s+", " ", text).upper()
    return re.sub(r"\s+", " ", text).strip(" ,")


def _npwp_digits(text: str) -> Optional[str]:
    match = re.search(r"(?:\d[\d.\-\s]{8,}\d)", text or "")
    if match is None:
        return None
    digits = re.sub(r"\D", "", match.group())
    if len(digits) in {15, 16}:
        return digits
    return None
