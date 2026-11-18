# -*- encoding: utf-8 -*-
"""Offline Indonesian province and kabupaten/kota codes for KTP parsing.

The table is a compact extract of official Kemendagri region codes (2-digit
province, 4-digit kabupaten/kota as they appear in a NIK). Display names
match the spellings already used on KTP fields (DKI JAKARTA, SLEMAN).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

_DATA_PATH = Path(__file__).with_name("ktp_wilayah.json")

# Extra compact aliases that are common on cards or in OCR.
_PROVINCE_ALIASES = {
    "DKIJAKARTA": "DKI JAKARTA",
    "DKI": "DKI JAKARTA",
    "DAERAHKHUSUSIBUKOTAJAKARTA": "DKI JAKARTA",
    "DIYOGYAKARTA": "DAERAH ISTIMEWA YOGYAKARTA",
    "DIY": "DAERAH ISTIMEWA YOGYAKARTA",
    "DAERAHISTIMEWAYOGYAKARTA": "DAERAH ISTIMEWA YOGYAKARTA",
}


@lru_cache(maxsize=1)
def _data() -> dict:
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def provinces() -> dict[str, str]:
    return dict(_data()["provinces"])


def regencies() -> dict[str, str]:
    return dict(_data()["regencies"])


def province_names() -> tuple[str, ...]:
    return tuple(sorted(set(provinces().values())))


def regency_names() -> tuple[str, ...]:
    return tuple(sorted(set(regencies().values())))


def lookup_province_code(code: str) -> Optional[str]:
    return provinces().get((code or "")[:2])


def lookup_regency_code(code: str) -> Optional[str]:
    digits = re.sub(r"\D", "", code or "")
    if len(digits) >= 4:
        return regencies().get(digits[:4])
    return None


def nik_region(nik: str) -> tuple[Optional[str], Optional[str]]:
    digits = re.sub(r"\D", "", nik or "")
    if len(digits) < 4:
        return None, None
    return lookup_province_code(digits[:2]), lookup_regency_code(digits[:4])


def _compact(text: str) -> str:
    return re.sub(r"[^A-Z]", "", (text or "").upper())


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, start=1):
        current = [index]
        for other_index, other in enumerate(right, start=1):
            insert = current[other_index - 1] + 1
            delete = previous[other_index] + 1
            replace = previous[other_index - 1] + (char != other)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def match_region(value: str, names: tuple[str, ...]) -> Optional[str]:
    """Fuzzy-match a province or kabupaten string, including truncated typos."""
    query = _compact(value)
    if len(query) < 3:
        return None
    if query in _PROVINCE_ALIASES:
        return _PROVINCE_ALIASES[query]
    by_compact = {_compact(name): name for name in names}
    if query in by_compact:
        return by_compact[query]

    best: list[tuple[int, int, str]] = []
    for compact, name in by_compact.items():
        options = [_levenshtein(query, compact)]
        if len(query) >= 8 and len(compact) >= 8:
            options.append(_levenshtein(query, compact[: len(query)]))
            options.append(_levenshtein(compact, query[: len(compact)]))
        distance = min(options)
        longest = max(len(query), len(compact))
        shortest = min(len(query), len(compact))
        if shortest <= 4:
            limit = 0
        elif longest <= 8:
            limit = 1
        elif longest <= 14:
            limit = 2
        else:
            limit = 3
        if distance <= limit:
            best.append((distance, -len(compact), name))
    if not best:
        return None
    best.sort()
    top = [item for item in best if item[0] == best[0][0]]
    unique = {item[2] for item in top}
    if len(unique) == 1:
        return best[0][2]
    return None
