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

from .ktp_lexicon import (
    AGAMA_CANON,
    GOL_DARAH_CANON,
    NAME_LEXICON,
    PEKERJAAN_ALIAS,
    PEKERJAAN_CANON,
    PLACE_GAZETTEER,
    PLACE_PREFIXES,
    PLACE_SUFFIXES,
)

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
    _recover_alamat(fields, lines, used)
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


def _closest_field(
    compact: str, *, loose: bool = False
) -> Optional[tuple[str, int]]:
    if len(compact) < 3:
        return None

    mapped = _VARIANT_TO_FIELD.get(compact)
    if mapped is not None:
        return mapped, 0

    best: Optional[tuple[int, int, str]] = None
    for field, canon in _CANON_COMPACT:
        limit = _edit_limit(len(canon))
        if loose and 3 <= len(canon) <= 6:
            limit = max(limit, 1)
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


def _is_label_typo(text: str) -> bool:
    """True when a short line is a misspelled field label rather than a value."""
    compact = _compact(text)
    if not (3 <= len(compact) <= 8):
        return False
    return _closest_field(compact, loose=True) is not None


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
    if text in {"--", "—"}:
        text = "-"
    if text == "0":
        text = "O"
    if text in GOL_DARAH_CANON:
        return text
    return None


def _is_subsequence(short: str, long: str) -> bool:
    index = 0
    for char in long:
        if index < len(short) and char == short[index]:
            index += 1
    return index == len(short)


def _match_catalog(
    value: str,
    names: Sequence[str],
    aliases: Optional[Mapping[str, str]] = None,
    *,
    loose: bool = False,
) -> Optional[str]:
    """Return the catalog spelling closest to ``value``, or None."""
    query = re.sub(r"[^A-Z]", "", _normalize_spacing(value).upper())
    if len(query) < 3:
        return None
    by_compact: dict[str, str] = {}
    for name in names:
        compact = re.sub(r"[^A-Z]", "", name.upper())
        if compact:
            by_compact[compact] = name.upper()
    if aliases:
        for alias, official in aliases.items():
            by_compact[re.sub(r"[^A-Z]", "", alias.upper())] = official.upper()

    if query in by_compact:
        return by_compact[query]

    best: list[tuple[int, int, str]] = []
    for compact, name in by_compact.items():
        limit = 1 if max(len(query), len(compact)) <= 8 else 2
        if loose:
            limit = max(limit, 2)
        if abs(len(query) - len(compact)) > limit:
            continue
        distance = _levenshtein(query, compact)
        if distance <= limit:
            best.append((distance, len(compact), name))
    if best:
        best.sort()
        top = [item for item in best if item[0] == best[0][0]]
        if len({item[2] for item in top}) == 1:
            return best[0][2]

    # Prefix / stump: "ISA" -> ISLAM, when it uniquely prefixes one entry.
    if len(query) >= 3:
        prefixed = [
            name
            for compact, name in by_compact.items()
            if compact.startswith(query) or query.startswith(compact)
        ]
        unique = set(prefixed)
        if len(unique) == 1:
            return next(iter(unique))

    # Dropped letters (often vowels): "TRHRO" is a subsequence of "TRIHARJO".
    if len(query) >= 5:
        dropped: list[tuple[int, int, str]] = []
        for compact, name in by_compact.items():
            gap = len(compact) - len(query)
            if 1 <= gap <= 3 and _is_subsequence(query, compact):
                dropped.append((gap, len(compact), name))
        if dropped:
            dropped.sort()
            if len(dropped) == 1 or dropped[0][0] < dropped[1][0]:
                return dropped[0][2]
    return None


_ADDRESS_PREFIXES = PLACE_PREFIXES + (
    "KP",
    "JL",
    "JLN",
    "GG",
    "DS",
    "DK",
    "DSN",
    "LRG",
    "PERUM",
    "KOMP",
)


def _correct_place_name(value: str, *, whole: bool = True) -> str:
    text = _expand_glued_place(value)
    if not text:
        return text
    parts = text.split()
    for index, prefix in enumerate(parts):
        bare = prefix.rstrip(".")
        if bare in _ADDRESS_PREFIXES and index + 1 < len(parts):
            tail = " ".join(parts[index + 1 :])
            fixed = _match_catalog(tail, PLACE_GAZETTEER)
            if fixed:
                return " ".join(parts[: index + 1] + [fixed])
            return text
    if not whole:
        return text
    return _match_catalog(text, PLACE_GAZETTEER) or text


def _clean_agama(value: str) -> Optional[str]:
    text = _normalize_spacing(value).upper()
    if not text:
        return None
    # Official catalog uses BUDDHA; "BUDHA" is a frequent OCR/spelling variant.
    aliases = {"BUDHA": "BUDDHA"}
    matched = _match_catalog(text, AGAMA_CANON, aliases, loose=True)
    return matched or text


def _clean_pekerjaan(value: str) -> Optional[str]:
    text = _normalize_spacing(value).upper()
    if not text:
        return None
    matched = _match_catalog(text, PEKERJAAN_CANON, PEKERJAAN_ALIAS)
    return matched or text


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
    if key == "agama":
        return _clean_agama(value)
    if key == "pekerjaan":
        return _clean_pekerjaan(value)
    if key in {"kel_desa", "kecamatan"}:
        text = _correct_place_name(value, whole=True)
        return text or None
    if key == "alamat":
        text = _correct_place_name(value, whole=False)
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
        if (
            field == "kel_desa"
            and _compact(match.group()) == "desa"
            and value
        ):
            # "DESA PETANJUNGAN" is an address line, not a Kel/Desa label.
            return []
        segments.append((field, value))
    return segments


def _fuzzy_leading_label(line: str) -> Optional[tuple[str, str]]:
    if re.search(r"[:：]", line):
        label, value = re.split(r"[:：]", line, maxsplit=1)
        if _DATE_RE.search(label):
            return None
        found = _closest_field(_compact(label), loose=True)
        if found is None:
            return None
        field, _distance = found
        remainder = _strip_leading_colon(value).strip()
        if field == "kel_desa" and _compact(label) == "desa" and remainder:
            return None
        return field, remainder

    tokens = line.split()
    best: Optional[tuple[int, int, str, str]] = None
    for end in range(1, len(tokens) + 1):
        chunk = " ".join(tokens[:end])
        if _DATE_RE.search(chunk):
            break
        loose = end == len(tokens) and 3 <= len(_compact(chunk)) <= 6
        found = _closest_field(_compact(chunk), loose=loose)
        if found is None:
            continue
        field, distance = found
        remainder = " ".join(tokens[end:]).strip()
        if field == "kel_desa" and _compact(chunk) == "desa" and remainder:
            continue
        candidate = (distance, end, field, remainder)
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


# Fields whose values come from a small printed vocabulary. A following line
# that does not clean to a catalog entry is left unused so another field can
# claim it (for example an address sitting under an empty Gol. Darah row).
_CLOSED_FIELDS = frozenset(
    {
        "gol_darah",
        "agama",
        "jenis_kelamin",
        "status_perkawinan",
        "kewarganegaraan",
    }
)


def _assign(fields: dict[str, Optional[str]], key: str, value: str) -> bool:
    cleaned = _clean_field(key, value)
    if cleaned:
        fields[key] = cleaned
        return True
    return False


def _take_nik(fields: dict[str, Optional[str]], lines: Sequence[str]) -> None:
    for line in lines:
        nik = find_nik(line)
        if nik:
            fields["nik"] = nik
            return


def _is_ttl_stump(line: str) -> bool:
    """A short 'tem…' token that is not itself a known place name."""
    compact = _compact(line)
    if not compact.startswith("tem") or not (4 <= len(compact) <= 12):
        return False
    place = re.sub(r"[^A-Z]", "", _normalize_spacing(line).upper())
    known = {re.sub(r"[^A-Z]", "", name.upper()) for name in PLACE_GAZETTEER}
    return place not in known


def _line_place_date(line: str) -> Optional[str]:
    return _clean_ttl(_strip_leading_colon(line)) if _DATE_RE.search(line or "") else None


def _consume_ttl_stump(
    fields: dict[str, Optional[str]],
    lines: Sequence[str],
    index: int,
    used: set[int],
) -> int:
    """If this line is a mangled Tempat/Tgl Lahir label, take the next date line."""
    if not _is_ttl_stump(lines[index]):
        return 0
    if index + 1 >= len(lines) or _is_structural(lines[index + 1]):
        return 0
    if not _line_place_date(lines[index + 1]):
        return 0
    if _assign(fields, "tempat_tgl_lahir", lines[index + 1]):
        used.update((index, index + 1))
        return 2
    return 0


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
        return _consume_ttl_stump(fields, lines, index, used)

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
        accepted = _assign(fields, key, value)
        if not accepted and key in _CLOSED_FIELDS:
            used.add(index)
            return 1
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
    if _is_label_typo(text) or _closest_field(_compact(text), loose=True) is not None:
        return False
    compact = _compact(text)
    if compact in {_compact(phrase) for phrase in _NOT_PLACE}:
        return False
    if _is_ttl_stump(text):
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


def _looks_like_address_line(line: str) -> bool:
    text = _strip_leading_colon(_squash_space(line))
    if not text or _DATE_RE.search(text) or _match_header(text):
        return False
    if _is_label_typo(text):
        return False
    compact = _compact(text)
    if compact in {_compact(phrase) for phrase in _NOT_PLACE}:
        return False
    if _match_catalog(text, AGAMA_CANON) or _match_catalog(text, PEKERJAAN_CANON, PEKERJAAN_ALIAS):
        return False
    return len(re.sub(r"[^a-z]", "", _fold(text))) >= 3


def _is_issue_place_line(lines: Sequence[str], index: int) -> bool:
    """True when this leftover place is the issue block (followed by a lone date)."""
    if index + 1 >= len(lines):
        return False
    nxt = _normalize_spacing(lines[index + 1])
    return _LONE_DATE_RE.match(nxt) is not None


def _recover_alamat(
    fields: dict[str, Optional[str]], lines: Sequence[str], used: set[int]
) -> None:
    """Fill an empty alamat from a leftover line next to the Alamat label."""
    if fields["alamat"]:
        return

    label_at: Optional[int] = None
    for index, line in enumerate(lines):
        segments = _split_labels(line)
        if len(segments) == 1 and segments[0][0] == "alamat":
            label_at = index
            break
    if label_at is None:
        return

    candidates: list[int] = []
    if label_at - 1 >= 0 and (label_at - 1) not in used:
        candidates.append(label_at - 1)
    if label_at + 1 < len(lines) and (label_at + 1) not in used:
        candidates.append(label_at + 1)

    for index in candidates:
        if _is_issue_place_line(lines, index):
            continue
        if not _looks_like_address_line(lines[index]):
            continue
        if _assign(fields, "alamat", lines[index]):
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
