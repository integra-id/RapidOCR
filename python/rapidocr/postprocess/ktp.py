# -*- encoding: utf-8 -*-
"""Parse Indonesian KTP (resident ID card) fields from OCR lines.

RapidOCR returns one string per text line. On a KTP those lines often glue
the label to the value, drop spaces inside a name, and keep a leading colon.
``parse_ktp`` maps that raw reading order into a fixed field dict. Other
languages and the OCR engine itself are unchanged.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional, Sequence

from .ktp_lexicon import NAME_LEXICON

KTP_FIELDS = (
    "nik",
    "nama",
    "tempat_tgl_lahir",
    "jenis_kelamin",
    "gol_darah",
    "alamat",
    "rt_rw",
    "kel_desa",
    "kecamatan",
    "agama",
    "status_perkawinan",
    "pekerjaan",
    "kewarganegaraan",
    "berlaku_hingga",
    "provinsi",
    "kabupaten_kota",
    "issued_place",
    "issued_date",
)

# Longer / more specific labels must be tried before short ones.
_FIELD_PATTERNS = (
    ("tempat_tgl_lahir", r"tempat\s*/?\s*tgl\s*lahir|t\.?\s*t\.?\s*l"),
    ("jenis_kelamin", r"jenis\s*kelamin"),
    ("status_perkawinan", r"status\s*perkawinan"),
    ("kewarganegaraan", r"kewarganegaraan"),
    ("berlaku_hingga", r"berlaku\s*(?:hingga|sampai)"),
    ("gol_darah", r"gol(?:ongan)?\.?\s*darah"),
    ("kel_desa", r"kel(?:urahan)?\s*/?\s*desa|kelurahan|desa"),
    ("kecamatan", r"kecamatan"),
    ("pekerjaan", r"pekerjaan"),
    ("rt_rw", r"rt\s*/?\s*rw"),
    ("alamat", r"alamat"),
    ("agama", r"agama"),
    ("nama", r"nama"),
    ("nik", r"nik"),
)

_LABEL_RE = re.compile(
    r"(?i)(?<![a-z])("
    + "|".join(f"(?P<{field}>{pattern})" for field, pattern in _FIELD_PATTERNS)
    + r")(?![a-z])"
)
_HEADER_RE = re.compile(r"(?i)^\s*(provinsi|kabupaten|kota)\s*(.*)$")
_GELAR = r"(?:(?:dr|drs|ir|prof|hj|h)\.\s*)?(?:[A-Za-z]{1,3}\.)+[A-Za-z]{1,4}\.?"
_GELAR_LINE_RE = re.compile(rf"(?i)^(?:{_GELAR})$")
_GELAR_TAIL_RE = re.compile(rf"(?i)(?:,|\s)\s*({_GELAR})\s*$")
_ADDR_ABBREV_RE = re.compile(r"(?i)\b(KP|JL|JLN|GG|DS|DK|DSN|LRG|PERUM|KOMP)\.(?=\S)")
_DATE_RE = re.compile(r"\d{2}[-/]\d{2}[-/]\d{4}")
_LONE_DATE_RE = re.compile(r"^(\d{2}[-/]\d{2}[-/]\d{4})$")
_PLACE_DATE_RE = re.compile(r"^(.+?),\s*(\d{2}[-/]\d{2}[-/]\d{4})$")
_PLACE_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]{1,40}$")

_GLUED_PLACES = {
    "JAWABARAT": "JAWA BARAT",
    "JAWATENGAH": "JAWA TENGAH",
    "JAWATIMUR": "JAWA TIMUR",
    "SUMATERAUTARA": "SUMATERA UTARA",
    "SUMATERABARAT": "SUMATERA BARAT",
    "SUMATERASELATAN": "SUMATERA SELATAN",
    "KALIMANTANBARAT": "KALIMANTAN BARAT",
    "KALIMANTANTENGAH": "KALIMANTAN TENGAH",
    "KALIMANTANTIMUR": "KALIMANTAN TIMUR",
    "KALIMANTANSELATAN": "KALIMANTAN SELATAN",
    "KALIMANTANUTARA": "KALIMANTAN UTARA",
    "SULAWESIUTARA": "SULAWESI UTARA",
    "SULAWESITENGAH": "SULAWESI TENGAH",
    "SULAWESISELATAN": "SULAWESI SELATAN",
    "SULAWESITENGGARA": "SULAWESI TENGGARA",
    "SULAWESIBARAT": "SULAWESI BARAT",
    "NUSATENGGARABARAT": "NUSA TENGGARA BARAT",
    "NUSATENGGARATIMUR": "NUSA TENGGARA TIMUR",
    "DKIJAKARTA": "DKI JAKARTA",
    "DIYOGYAKARTA": "DI YOGYAKARTA",
    "KEPULAUANBANGKABELITUNG": "KEPULAUAN BANGKA BELITUNG",
    "KEPULAUANRIAU": "KEPULAUAN RIAU",
    "PAPUABARAT": "PAPUA BARAT",
    "PAPUABARATDAYA": "PAPUA BARAT DAYA",
    "PAPUASELATAN": "PAPUA SELATAN",
    "PAPUATENGAH": "PAPUA TENGAH",
    "PAPUAPEGUNUNGAN": "PAPUA PEGUNUNGAN",
}


def parse_ktp(
    source: Any, boxes: Optional[Sequence[Any]] = None
) -> dict[str, Optional[str]]:
    """Map OCR lines to KTP fields.

    ``source`` may be a sequence of strings or a RapidOCR result with
    ``txts`` and optional ``boxes``. ``boxes`` follow RapidOCR order: one
    4-point polygon, or ``[x1, y1, x2, y2]``, per line. Missing fields are
    ``None``.
    """
    lines, boxes = _coerce_input(source, boxes)
    if boxes is not None and len(boxes) == len(lines):
        order = sorted(
            range(len(lines)), key=lambda index: _box_reading_key(boxes[index])
        )
        lines = [lines[index] for index in order]

    lines = [_squash_space(line) for line in lines if _squash_space(line)]
    fields: dict[str, Optional[str]] = {key: None for key in KTP_FIELDS}
    _take_nik(fields, lines)

    used: set[int] = set()
    index = 0
    while index < len(lines):
        consumed = _consume_at(fields, lines, index, used)
        index += consumed if consumed else 1

    _assign_issued(fields, lines, used)
    return fields


def is_valid_nik(nik: str) -> bool:
    """Return True when ``nik`` is 16 digits with a plausible KTP birth date."""
    if not re.fullmatch(r"\d{16}", nik or ""):
        return False

    day = int(nik[6:8])
    month = int(nik[8:10])
    if day > 40:
        day -= 40
    return 1 <= day <= 31 and 1 <= month <= 12


def find_nik(text: str) -> Optional[str]:
    """Return the first valid 16-digit NIK in ``text``, ignoring separators."""
    compact = re.sub(r"(?<=\d)[\s-]+(?=\d)", "", text or "")
    for match in re.finditer(r"\d{16}", compact):
        candidate = match.group()
        if is_valid_nik(candidate):
            return candidate
    return None


def _coerce_input(
    source: Any, boxes: Optional[Sequence[Any]]
) -> tuple[list[str], Optional[list[Any]]]:
    if hasattr(source, "txts") or (
        not isinstance(source, (str, bytes)) and hasattr(source, "boxes")
    ):
        raw_lines = getattr(source, "txts", None) or ()
        if boxes is None:
            boxes = getattr(source, "boxes", None)
    elif isinstance(source, (str, bytes)):
        raw_lines = [source]
    else:
        raw_lines = source or ()

    lines = ["" if line is None else str(line) for line in raw_lines]
    box_list = _as_box_list(boxes, len(lines))
    return lines, box_list


def _as_box_list(boxes: Optional[Sequence[Any]], count: int) -> Optional[list[Any]]:
    if boxes is None:
        return None
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    try:
        box_list = list(boxes)
    except TypeError:
        return None
    if len(box_list) != count:
        return None
    return box_list


def _box_reading_key(box: Any) -> tuple[float, float]:
    if hasattr(box, "tolist"):
        box = box.tolist()
    if (
        isinstance(box, Sequence)
        and len(box) == 4
        and not isinstance(box[0], (Sequence, Mapping))
    ):
        return float(box[1]), float(box[0])

    points = list(box)
    ys = [float(point[1]) for point in points]
    xs = [float(point[0]) for point in points]
    return min(ys), min(xs)


def _squash_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _strip_leading_colon(text: str) -> str:
    return re.sub(r"^[\s:：]+", "", text).strip()


def _normalize_spacing(text: str) -> str:
    cleaned = _strip_leading_colon(_squash_space(text))
    cleaned = _ADDR_ABBREV_RE.sub(lambda match: match.group(1).upper() + ". ", cleaned)
    cleaned = re.sub(r",(?=\S)", ", ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _expand_glued_place(value: str) -> str:
    upper = _normalize_spacing(value).upper()
    compact = re.sub(r"[^A-Z]", "", upper)
    return _GLUED_PLACES.get(compact, upper)


def _segment_token(token: str) -> str:
    word = token.upper()
    if word in NAME_LEXICON:
        return word

    length = len(word)
    best: list[Optional[tuple[int, list[str]]]] = [None] * (length + 1)
    best[0] = (0, [])
    for start in range(length):
        current = best[start]
        if current is None:
            continue
        for end in range(start + 1, length + 1):
            piece = word[start:end]
            if piece not in NAME_LEXICON:
                continue
            score = current[0] + len(piece) ** 2
            candidate = (score, current[1] + [piece])
            if best[end] is None or score > best[end][0]:
                best[end] = candidate

    if best[length] is None:
        return token
    return " ".join(best[length][1])


def _segment_name_text(text: str) -> str:
    spaced = re.sub(
        r"[A-Z]{8,}", lambda match: _segment_token(match.group(0)), text.upper()
    )
    return re.sub(r"\s+", " ", spaced).strip()


def _peel_gelar(value: str) -> tuple[str, Optional[str]]:
    match = _GELAR_TAIL_RE.search(value)
    if match is None:
        return value, None
    return value[: match.start()].strip(" ,"), match.group(1).strip()


def _canon_gelar(gelar: str) -> str:
    parts = [part for part in gelar.split(".") if part]
    titled = [
        part[0].upper() + part[1:].lower() if part.isalpha() else part for part in parts
    ]
    return ".".join(titled)


def _clean_nama(value: str) -> Optional[str]:
    text = _normalize_spacing(value)
    if not text:
        return None
    body, gelar = _peel_gelar(text)
    body = _segment_name_text(body.strip(" ,"))
    if not body:
        return None
    if gelar:
        return f"{body}, {_canon_gelar(gelar)}"
    return body


def _clean_field(key: str, value: str) -> Optional[str]:
    if key == "nik":
        return find_nik(value)
    if key == "nama":
        return _clean_nama(value)
    if key == "jenis_kelamin":
        text = _normalize_spacing(value).upper().replace("LAKI LAKI", "LAKI-LAKI")
        return text or None
    if key == "gol_darah":
        text = _normalize_spacing(value).upper().replace(" ", "")
        if text in {"-", "--", "—"}:
            return "-"
        return text or None
    if key in {"provinsi", "kabupaten_kota"}:
        text = _expand_glued_place(value)
        return text or None

    text = _normalize_spacing(value).upper()
    return text or None


def _match_header(line: str) -> Optional[tuple[str, str]]:
    match = _HEADER_RE.match(line)
    if match is None:
        return None
    kind = match.group(1).lower()
    key = "provinsi" if kind == "provinsi" else "kabupaten_kota"
    return key, match.group(2).strip()


def _split_labels(line: str) -> list[tuple[str, str]]:
    matches = list(_LABEL_RE.finditer(line))
    if not matches:
        return []
    if line[: matches[0].start()].strip(" :") != "":
        return []

    segments: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        value = _strip_leading_colon(line[match.end() : end]).strip()
        field = next(
            (
                name
                for name, captured in match.groupdict().items()
                if captured is not None
            ),
            None,
        )
        if field is None:
            continue
        segments.append((field, value))
    return segments


def _is_gelar_line(line: str) -> bool:
    return _GELAR_LINE_RE.fullmatch(line.strip()) is not None


def _is_structural(line: str) -> bool:
    return _match_header(line) is not None or bool(_split_labels(line))


def _merge_gelar(name: str, gelar: str) -> str:
    return f"{name.rstrip(' ,')}, {gelar.strip()}"


def _assign(fields: dict[str, Optional[str]], key: str, value: str) -> None:
    cleaned = _clean_field(key, value)
    if cleaned:
        fields[key] = cleaned


def _take_nik(fields: dict[str, Optional[str]], lines: Sequence[str]) -> None:
    for line in lines:
        nik = find_nik(line)
        if nik:
            fields["nik"] = nik
            return


def _consume_at(
    fields: dict[str, Optional[str]], lines: Sequence[str], index: int, used: set[int]
) -> int:
    line = lines[index]
    header = _match_header(line)
    if header is not None:
        key, value = header
        if value:
            _assign(fields, key, value)
            used.add(index)
            return 1
        if index + 1 < len(lines) and not _is_structural(lines[index + 1]):
            _assign(fields, key, lines[index + 1])
            used.update((index, index + 1))
            return 2
        used.add(index)
        return 1

    segments = _split_labels(line)
    if not segments:
        return 0

    if len(segments) == 1 and not segments[0][1]:
        key = segments[0][0]
        if index + 1 >= len(lines) or _is_structural(lines[index + 1]):
            used.add(index)
            return 1
        value = lines[index + 1]
        step = 2
        if (
            key == "nama"
            and index + 2 < len(lines)
            and _is_gelar_line(lines[index + 2])
        ):
            value = _merge_gelar(value, lines[index + 2])
            step = 3
        _assign(fields, key, value)
        used.update(range(index, index + step))
        return step

    step = 1
    for key, value in segments:
        if (
            key == "nama"
            and value
            and index + 1 < len(lines)
            and _is_gelar_line(lines[index + 1])
            and _peel_gelar(_normalize_spacing(value))[1] is None
        ):
            value = _merge_gelar(value, lines[index + 1])
            step = 2
        if value:
            _assign(fields, key, value)
    used.update(range(index, index + step))
    return step


def _norm_date(value: str) -> str:
    return value.replace("/", "-")


def _date_in(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    match = _DATE_RE.search(value)
    if match is None:
        return None
    return _norm_date(match.group())


def _assign_issued(
    fields: dict[str, Optional[str]], lines: Sequence[str], used: Iterable[int]
) -> None:
    if fields["issued_date"] and fields["issued_place"]:
        return

    used_indexes = set(used)
    birth = _date_in(fields.get("tempat_tgl_lahir"))
    leftovers: list[str] = []
    for index, line in enumerate(lines):
        if index in used_indexes:
            continue
        text = _normalize_spacing(line)
        if not text or text in {"-", "--", "—"}:
            continue
        if _is_structural(text) or _is_gelar_line(text):
            continue
        leftovers.append(text)

    for index, text in enumerate(leftovers):
        place_date = _PLACE_DATE_RE.match(text)
        if place_date is not None:
            date = _norm_date(place_date.group(2))
            if date == birth:
                continue
            fields["issued_place"] = (
                fields["issued_place"] or place_date.group(1).upper()
            )
            fields["issued_date"] = fields["issued_date"] or date
            continue

        lone_date = _LONE_DATE_RE.match(text)
        if lone_date is None:
            continue
        date = _norm_date(lone_date.group(1))
        if date == birth:
            continue
        fields["issued_date"] = fields["issued_date"] or date
        if fields["issued_place"] is None and index > 0:
            previous = leftovers[index - 1]
            if _PLACE_LINE_RE.fullmatch(previous):
                fields["issued_place"] = previous.upper()
