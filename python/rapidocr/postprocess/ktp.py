# -*- encoding: utf-8 -*-
"""Parse Indonesian KTP (resident ID card) fields from OCR lines.

RapidOCR returns one string per text line. On a KTP those lines often glue
the label to the value, drop spaces inside a name, and keep a leading colon.
``parse_ktp`` maps that raw reading order into a fixed field dict. Other
languages and the OCR engine itself are unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Mapping, Optional, Sequence

from .ktp_lexicon import NAME_LEXICON, PLACE_PREFIXES, PLACE_SUFFIXES

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
_PLACE_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]{1,40}$")
_NOT_PLACE = {
    "SEUMUR HIDUP",
    "WNI",
    "WNA",
    "KAWIN",
    "BELUM KAWIN",
    "CERAI HIDUP",
    "CERAI MATI",
    "ISLAM",
    "KRISTEN",
    "KATOLIK",
    "HINDU",
    "BUDDHA",
    "KONGHUCU",
    "LAKI-LAKI",
    "LAKILAKI",
    "PEREMPUAN",
}

# Canonical spellings plus labels seen on phone photos. Matching uses the
# compact form (case, accents, spaces, and punctuation removed).
_LABEL_SPELLINGS = (
    ("nik", ("nik",)),
    ("nama", ("nama", "namá")),
    (
        "tempat_tgl_lahir",
        ("tempat/tgl lahir", "ttl", "tempai/tgilahir", "tempot/tgtlahr"),
    ),
    ("jenis_kelamin", ("jenis kelamin", "jens kelamin")),
    ("gol_darah", ("gol darah", "gol. darah", "golongan darah", "goldarah")),
    ("alamat", ("alamat",)),
    ("rt_rw", ("rt/rw", "rt rw", "rtrw")),
    (
        "kel_desa",
        ("kel/desa", "kelurahan/desa", "kelurahan", "desa", "ke/desa", "keidesa"),
    ),
    ("kecamatan", ("kecamatan", "kacamatan")),
    ("agama", ("agama",)),
    ("status_perkawinan", ("status perkawinan", "slatus perkawinan")),
    ("pekerjaan", ("pekerjaan",)),
    ("kewarganegaraan", ("kewarganegaraan",)),
    ("berlaku_hingga", ("berlaku hingga", "berlaku sampai", "berlakuhingga")),
)
# Short labels stay exact. Longer ones tolerate a few OCR edits.
_CANON_COMPACT = (
    ("nik", "nik"),
    ("nama", "nama"),
    ("tempat_tgl_lahir", "tempattgllahir"),
    ("jenis_kelamin", "jeniskelamin"),
    ("gol_darah", "goldarah"),
    ("gol_darah", "golongandarah"),
    ("alamat", "alamat"),
    ("rt_rw", "rtrw"),
    ("kel_desa", "keldesa"),
    ("kel_desa", "kelurahan"),
    ("kel_desa", "kelurahandesa"),
    ("kecamatan", "kecamatan"),
    ("agama", "agama"),
    ("status_perkawinan", "statusperkawinan"),
    ("pekerjaan", "pekerjaan"),
    ("kewarganegaraan", "kewarganegaraan"),
    ("berlaku_hingga", "berlakuhingga"),
    ("berlaku_hingga", "berlakusampai"),
)

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

    _recover_nama(fields, lines, used)
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


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_marks.lower()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _fold(text))


def _variant_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for field, spellings in _LABEL_SPELLINGS:
        for spelling in spellings:
            mapping[_compact(spelling)] = field
    return mapping


_VARIANT_TO_FIELD = _variant_map()


def _edit_limit(length: int) -> int:
    if length <= 4:
        return 0
    if length <= 8:
        return 1
    return 2


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


def _closest_field(compact: str) -> Optional[tuple[str, int]]:
    if len(compact) < 3:
        return None

    mapped = _VARIANT_TO_FIELD.get(compact)
    if mapped is not None:
        return mapped, 0

    best: Optional[tuple[int, int, str]] = None
    for field, canon in _CANON_COMPACT:
        limit = _edit_limit(len(canon))
        if abs(len(compact) - len(canon)) > limit:
            continue
        distance = _levenshtein(compact, canon)
        if distance > limit:
            continue
        rank = (distance, -len(canon), field)
        if best is None or rank < best:
            best = rank
    if best is None:
        return None
    return best[2], best[0]


def _normalize_spacing(text: str) -> str:
    cleaned = _strip_leading_colon(_squash_space(text))
    cleaned = _ADDR_ABBREV_RE.sub(lambda match: match.group(1).upper() + ". ", cleaned)
    cleaned = re.sub(r",(?=\S)", ", ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _split_place_token(token: str) -> str:
    if token in _GLUED_PLACES:
        return _GLUED_PLACES[token]

    for prefix in PLACE_PREFIXES:
        if token.startswith(prefix) and len(token) >= len(prefix) + 3:
            rest = _split_place_token(token[len(prefix) :])
            return f"{prefix} {rest}"

    for suffix in PLACE_SUFFIXES:
        if token.endswith(suffix) and len(token) >= len(suffix) + 3:
            head = _split_place_token(token[: -len(suffix)])
            return f"{head} {suffix}"

    return token


def _restore_place_spacing(text: str) -> str:
    spaced = re.sub(
        r"[A-Z]{8,}", lambda match: _split_place_token(match.group(0)), text
    )
    return re.sub(r"\s+", " ", spaced).strip()


def _expand_glued_place(value: str) -> str:
    upper = _normalize_spacing(value).upper()
    compact = re.sub(r"[^A-Z]", "", upper)
    if compact in _GLUED_PLACES:
        return _GLUED_PLACES[compact]
    return _restore_place_spacing(upper)


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


def _parse_place_date(text: str) -> Optional[tuple[str, str]]:
    match = re.match(r"^(.+?)[,.]\s*(\d{2}[-/]\d{2}[-/]\d{4})$", text.strip())
    if match is None:
        return None
    place = match.group(1).strip(" ,.")
    if not re.search(r"[A-Za-z]", place):
        return None
    return place, _norm_date(match.group(2))


def _clean_ttl(value: str) -> Optional[str]:
    text = _normalize_spacing(value).upper()
    text = re.sub(r"\.(?=\d{2}[-/]\d{2}[-/]\d{4})", ", ", text)
    pair = _parse_place_date(text)
    if pair is not None:
        place, date = pair
        return f"{_expand_glued_place(place)}, {date}"
    text = _expand_glued_place(text)
    return text or None


def _clean_gender(value: str) -> Optional[str]:
    text = _normalize_spacing(value).upper()
    compact = re.sub(r"[^A-Z]", "", text)
    if compact == "LAKILAKI":
        return "LAKI-LAKI"
    if compact == "PEREMPUAN":
        return "PEREMPUAN"
    return text or None


def _clean_blood(value: str) -> Optional[str]:
    text = _normalize_spacing(value).upper().replace(" ", "")
    if text in {"-", "--", "—"}:
        return "-"
    if text in {"0", "O"}:
        return "O"
    return text or None


def _clean_field(key: str, value: str) -> Optional[str]:
    if key == "nik":
        return find_nik(value)
    if key == "nama":
        return _clean_nama(value)
    if key == "jenis_kelamin":
        return _clean_gender(value)
    if key == "gol_darah":
        return _clean_blood(value)
    if key == "tempat_tgl_lahir":
        return _clean_ttl(value)
    if key in {"provinsi", "kabupaten_kota", "alamat", "kel_desa", "kecamatan"}:
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


def _exact_label_segments(line: str) -> list[tuple[str, str]]:
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


def _fuzzy_leading_label(line: str) -> Optional[tuple[str, str]]:
    if re.search(r"[:：]", line):
        label, value = re.split(r"[:：]", line, maxsplit=1)
        if _DATE_RE.search(label):
            return None
        found = _closest_field(_compact(label))
        if found is None:
            return None
        return found[0], _strip_leading_colon(value).strip()

    tokens = line.split()
    best: Optional[tuple[int, int, str, str]] = None
    for end in range(1, len(tokens) + 1):
        chunk = " ".join(tokens[:end])
        if _DATE_RE.search(chunk):
            break
        found = _closest_field(_compact(chunk))
        if found is None:
            continue
        field, distance = found
        candidate = (distance, end, field, " ".join(tokens[end:]).strip())
        if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
            best = candidate
    if best is None:
        return None
    return best[2], best[3]


def _split_labels(line: str) -> list[tuple[str, str]]:
    exact = _exact_label_segments(line)
    if exact:
        return exact
    fuzzy = _fuzzy_leading_label(_strip_leading_colon(line))
    if fuzzy is None:
        return []
    return [fuzzy]


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


def _looks_like_name_line(line: str) -> bool:
    text = _strip_leading_colon(_squash_space(line))
    if not text or _DATE_RE.search(text) or _match_header(text):
        return False
    if _closest_field(_compact(text)) is not None:
        return False
    compact = _compact(text)
    if compact in {_compact(phrase) for phrase in _NOT_PLACE}:
        return False
    return len(re.sub(r"[^a-z]", "", _fold(text))) >= 3


def _recover_nama(
    fields: dict[str, Optional[str]], lines: Sequence[str], used: set[int]
) -> None:
    """If the nama label was unreadable, take the name-like line after NIK."""
    if fields["nama"]:
        return

    start = 0
    for index, line in enumerate(lines):
        if find_nik(line):
            start = index + 1
            break

    for index in range(start, len(lines)):
        if index in used or not _looks_like_name_line(lines[index]):
            continue
        _assign(fields, "nama", lines[index])
        if fields["nama"]:
            used.add(index)
            return


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

    # The issue block sits at the bottom of the card. An earlier place+date
    # line is the birth row and must not win when a later date exists.
    candidates: list[tuple[Optional[str], str]] = []
    for index, text in enumerate(leftovers):
        pair = _parse_place_date(text)
        if pair is not None:
            place, date = pair
            if date != birth:
                candidates.append((place, date))
            continue

        lone_date = _LONE_DATE_RE.match(text)
        if lone_date is None:
            continue
        date = _norm_date(lone_date.group(1))
        if date == birth:
            continue
        place = None
        if index > 0:
            previous = leftovers[index - 1].upper()
            if (
                _PLACE_LINE_RE.fullmatch(previous)
                and previous not in _NOT_PLACE
                and _parse_place_date(previous) is None
            ):
                place = previous
        candidates.append((place, date))

    if not candidates:
        return

    place, date = candidates[-1]
    if fields["issued_date"] is None:
        fields["issued_date"] = date
    if place and fields["issued_place"] is None:
        fields["issued_place"] = _expand_glued_place(place)
