# -*- encoding: utf-8 -*-
"""Peel and normalize academic or honorific titles on a KTP name."""

from __future__ import annotations

import re
from typing import Optional

from .ktp_lexicon import GELAR_CANON, GELAR_COMPACT, GELAR_PREFIXES

# Dotted titles may be glued to the last name token: KUSMARWANTOS.T.
_DOTTED_TAIL = re.compile(
    r"(?i)(?P<body>[A-Za-z]{2,})(?P<title>"
    r"S\.?\s*Pd\.?\s*I\.?"
    r"|S\.?\s*Kom\.?"
    r"|S\.?\s*Sos\.?"
    r"|S\.?\s*Ag\.?"
    r"|S\.?\s*Pd\.?"
    r"|S\.?\s*H\.?"
    r"|S\.?\s*E\.?"
    r"|S\.?\s*T\.?"
    r"|M\.?\s*Kom\.?"
    r"|M\.?\s*Si\.?"
    r"|M\.?\s*T\.?"
    r"|M\.?\s*M\.?"
    r"|A\.?\s*Md\.?"
    r"|Ph\.?\s*D\.?"
    r")$"
)


def _compact(text: str) -> str:
    return re.sub(r"[^A-Za-z]", "", text or "").upper()


def canon_title(text: str) -> Optional[str]:
    raw = (text or "").strip(" ,.;")
    if not raw:
        return None
    compact = _compact(raw)
    if compact in GELAR_COMPACT:
        canon = GELAR_COMPACT[compact]
        # Keep lowercase dr. only when the source was clearly lowercase.
        if canon == "Dr." and raw.startswith("dr") and not raw.startswith("Dr") and "DRS" not in compact:
            if compact == "DR" and raw[:2] == "dr":
                return "dr."
        return canon
    # Already a canonical spelling (maybe missing a trailing dot).
    for official in GELAR_CANON:
        if _compact(official) == compact:
            return official
    return None


def _is_title_token(token: str) -> Optional[str]:
    return canon_title(token)


def split_titles(value: str) -> tuple[str, list[str], list[str]]:
    """Return (bare name, prefix titles, suffix titles)."""
    text = re.sub(r"\s+", " ", (value or "").replace("，", ",").strip(" ,"))
    if not text:
        return "", [], []

    # Work on comma-separated chunks so "S.T., MBA" peels cleanly.
    tokens = [part.strip(" ,") for part in re.split(r"\s*,\s*|\s+", text) if part.strip(" ,")]
    prefixes: list[str] = []
    while tokens:
        title = _is_title_token(tokens[0])
        if title is None or title not in GELAR_PREFIXES and title not in {"Prof.", "Drs.", "Dra.", "Ir.", "Dr.", "dr.", "Hj.", "H."}:
            # Only honorifics at the front.
            if title is not None and title in {"Prof.", "Drs.", "Dra.", "Ir.", "Dr.", "dr.", "Hj.", "H."}:
                prefixes.append(title)
                tokens.pop(0)
                continue
            break
        if title in {"Prof.", "Drs.", "Dra.", "Ir.", "Dr.", "dr.", "Hj.", "H."}:
            prefixes.append(title)
            tokens.pop(0)
            continue
        break

    suffixes: list[str] = []
    while tokens:
        title = _is_title_token(tokens[-1])
        if title is None:
            glued = _DOTTED_TAIL.search(tokens[-1])
            if glued is None:
                break
            title = canon_title(glued.group("title"))
            body = glued.group("body")
            if not title:
                break
            tokens[-1] = body
            suffixes.insert(0, title)
            continue
        suffixes.insert(0, title)
        tokens.pop()

    bare = " ".join(tokens).strip(" ,")
    return bare, prefixes, suffixes


def format_nama(bare: str, prefixes: list[str], suffixes: list[str]) -> str:
    head = " ".join(prefixes + ([bare] if bare else [])).strip()
    if suffixes:
        return f"{head}, " + ", ".join(suffixes) if head else ", ".join(suffixes)
    return head


def format_gelar(prefixes: list[str], suffixes: list[str]) -> Optional[str]:
    titles = prefixes + suffixes
    return ", ".join(titles) if titles else None
