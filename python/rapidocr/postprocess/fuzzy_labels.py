# -*- encoding: utf-8 -*-
"""Small fuzzy label matcher shared by Indonesian document parsers."""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping, Optional, Sequence


def fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return stripped.lower()


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", fold(text))


def levenshtein(left: str, right: str) -> int:
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
            current.append(
                min(
                    current[other_index - 1] + 1,
                    previous[other_index] + 1,
                    previous[other_index - 1] + (char != other),
                )
            )
        previous = current
    return previous[-1]


def match_label(text: str, labels: Mapping[str, Sequence[str]]) -> Optional[str]:
    """Return the field whose label spelling is closest to ``text``."""
    token = compact(text)
    if len(token) < 3:
        return None
    best: Optional[tuple[int, int, str]] = None
    for field, spellings in labels.items():
        for spelling in spellings:
            canon = compact(spelling)
            if not canon:
                continue
            if token == canon:
                return field
            limit = 0 if len(canon) <= 4 else 1 if len(canon) <= 10 else 2
            if abs(len(token) - len(canon)) > limit:
                continue
            distance = levenshtein(token, canon)
            if distance > limit:
                continue
            rank = (distance, -len(canon), field)
            if best is None or rank < best:
                best = rank
    if best is None:
        return None
    return best[2]


def split_label(
    line: str, labels: Mapping[str, Sequence[str]]
) -> tuple[Optional[str], str]:
    raw = (line or "").strip()
    if ":" in raw or "：" in raw:
        label, value = re.split(r"[:：]", raw, maxsplit=1)
        field = match_label(label, labels)
        if field:
            return field, value.strip()
    field = match_label(raw, labels)
    if field:
        return field, ""
    return None, raw
