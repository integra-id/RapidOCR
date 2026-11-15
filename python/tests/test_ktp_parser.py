# -*- encoding: utf-8 -*-
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapidocr import parse_ktp as exported_parse_ktp
from rapidocr.postprocess import KTP_FIELDS, parse_ktp
from rapidocr.postprocess.ktp import find_nik, is_valid_nik

# Lines observed from a Garut KTP read with PP-OCRv6 small, lang_type=id.
# Spaces inside the name are gone, the title falls onto the next line, and
# several headers are glued to their values.
GARUT_KTP_LINES = [
    "PROVINSIJAWA BARAT",
    "KABUPATENGARUT",
    "NIK",
    "3205192301880001",
    "Nama",
    ":YOGIIRFANROSYADI,",
    "S.Pd.I",
    "Tempat/Tgl Lahir",
    ":GARUT,23-01-1988",
    "Alamat",
    ":KP.CIKANCUNG",
]


def test_garut_sample_returns_full_schema():
    parsed = parse_ktp(GARUT_KTP_LINES)

    assert tuple(parsed) == KTP_FIELDS
    assert parsed["nik"] == "3205192301880001"
    assert parsed["nama"] == "YOGI IRFAN ROSYADI, S.Pd.I"
    assert parsed["tempat_tgl_lahir"] == "GARUT, 23-01-1988"
    assert parsed["alamat"] == "KP. CIKANCUNG"
    assert parsed["provinsi"] == "JAWA BARAT"
    assert parsed["kabupaten_kota"] == "GARUT"
    assert parsed["jenis_kelamin"] is None
    assert parsed["issued_place"] is None
    assert parsed["issued_date"] is None


def test_same_line_name_keeps_title_and_strips_colon():
    parsed = parse_ktp(["Nama: :YOGIIRFANROSYADI, S.Pd.I"])

    assert parsed["nama"] == "YOGI IRFAN ROSYADI, S.Pd.I"


def test_nik_validation_accepts_female_day_offset_and_rejects_bad_dates():
    assert is_valid_nik("3205192301880001")
    assert is_valid_nik("3205194101880002")
    assert not is_valid_nik("3205199932880001")
    assert not is_valid_nik("320519230188000")
    assert find_nik("NIK 3205 1923 0188 0001") == "3205192301880001"
    assert find_nik("3205199932880001") is None


def test_remaining_fields_and_issue_block():
    lines = [
        "PROVINSI DKI JAKARTA",
        "KOTA JAKARTA PUSAT",
        "NIK : 3171014101900001",
        "Nama : SITIAMINAH",
        "Tempat/Tgl Lahir : JAKARTA, 01-01-1990",
        "Jenis Kelamin : PEREMPUAN Gol. Darah : AB",
        "Alamat : JL.MERDEKA",
        "RT/RW : 001/002",
        "Kel/Desa : SUKAMAJU",
        "Kecamatan : GAMBIR",
        "Agama : ISLAM",
        "Status Perkawinan : KAWIN",
        "Pekerjaan : GURU",
        "Kewarganegaraan : WNI",
        "Berlaku Hingga : SEUMUR HIDUP",
        "JAKARTA",
        "15-08-2019",
    ]

    parsed = parse_ktp(lines)

    assert parsed["nik"] == "3171014101900001"
    assert parsed["nama"] == "SITI AMINAH"
    assert parsed["tempat_tgl_lahir"] == "JAKARTA, 01-01-1990"
    assert parsed["jenis_kelamin"] == "PEREMPUAN"
    assert parsed["gol_darah"] == "AB"
    assert parsed["alamat"] == "JL. MERDEKA"
    assert parsed["rt_rw"] == "001/002"
    assert parsed["kel_desa"] == "SUKAMAJU"
    assert parsed["kecamatan"] == "GAMBIR"
    assert parsed["agama"] == "ISLAM"
    assert parsed["status_perkawinan"] == "KAWIN"
    assert parsed["pekerjaan"] == "GURU"
    assert parsed["kewarganegaraan"] == "WNI"
    assert parsed["berlaku_hingga"] == "SEUMUR HIDUP"
    assert parsed["provinsi"] == "DKI JAKARTA"
    assert parsed["kabupaten_kota"] == "JAKARTA PUSAT"
    assert parsed["issued_place"] == "JAKARTA"
    assert parsed["issued_date"] == "15-08-2019"


def test_unknown_glued_name_is_left_unchanged():
    parsed = parse_ktp(["Nama", ":QQQQQQQQ"])

    assert parsed["nama"] == "QQQQQQQQ"


def test_boxes_restore_reading_order():
    lines = [
        ":YOGIIRFANROSYADI,",
        "S.Pd.I",
        "Nama",
        "3205192301880001",
        "NIK",
    ]
    boxes = [
        [[10, 80], [200, 80], [200, 100], [10, 100]],
        [[10, 104], [80, 104], [80, 120], [10, 120]],
        [[10, 60], [80, 60], [80, 75], [10, 75]],
        [[10, 40], [220, 40], [220, 55], [10, 55]],
        [[10, 20], [40, 20], [40, 35], [10, 35]],
    ]

    parsed = parse_ktp(lines, boxes=boxes)

    assert parsed["nik"] == "3205192301880001"
    assert parsed["nama"] == "YOGI IRFAN ROSYADI, S.Pd.I"


def test_rapidocr_output_object_is_accepted():
    result = SimpleNamespace(txts=tuple(GARUT_KTP_LINES), boxes=None)

    parsed = parse_ktp(result)

    assert parsed["nik"] == "3205192301880001"
    assert parsed["nama"] == "YOGI IRFAN ROSYADI, S.Pd.I"


def test_parse_ktp_is_exported_from_package_root():
    assert exported_parse_ktp is parse_ktp


# Phone-photo OCR (txts only). The broken labels are the strings RapidOCR
# emitted; neighboring values are the raw lines for the fields under test.
PEMALANG_KTP_LINES = [
    "PROVINSIJAWA TENGAH",
    "KABUPATENPEMALANG",
    "NIK",
    "3327011112890001",
    "Namá",
    ":RISWANDI",
    "Tempai/TgiLahir",
    "PEMALANG, 11-12-1989",
    "Jens Kelamin",
    ":LAKILAKI",
    "Alamat",
    ":DESAPETANJUNGAN",
    "Ke/Desa",
    ":PETANJUNGAN",
    "Kacamatan",
    ":ULUJAMI",
    "Slatus Perkawinan: KAWIN",
    "BerlakuHingga",
    "SEUMUR HIDUP",
    "PEMALANG",
    "13-11-2018",
]

BEKASI_KTP_LINES = [
    "PROVINSI JAWA BARAT",
    "KABUPATEN BEKASI",
    "NIK",
    "3216010809780001",
    "Nama",
    ":ANDI WIBOWO",
    "Tempot/TgtLahr",
    "PEMALANG.08-09-1978",
    "Jens Kelamin",
    "LAKILAKI",
    "GolDarah",
    "O",
    "KeiDesa",
    "SUKAMAJU",
    "Kacamatan",
    "CIKARANGSELATAN",
    "Slatus Perkawinan: KAWIN",
    "BerlakuHingga",
    "SEUMUR HIDUP",
    "CIKARANG",
    "04-05-2016",
]


def test_pemalang_phone_ocr_lines():
    parsed = parse_ktp(PEMALANG_KTP_LINES)

    assert parsed["nik"] == "3327011112890001"
    assert parsed["nama"] == "RISWANDI"
    assert parsed["tempat_tgl_lahir"] == "PEMALANG, 11-12-1989"
    assert parsed["jenis_kelamin"] == "LAKI-LAKI"
    assert parsed["alamat"] == "DESA PETANJUNGAN"
    assert parsed["kel_desa"] == "PETANJUNGAN"
    assert parsed["kecamatan"] == "ULUJAMI"
    assert parsed["status_perkawinan"] == "KAWIN"
    assert parsed["berlaku_hingga"] == "SEUMUR HIDUP"
    assert parsed["provinsi"] == "JAWA TENGAH"
    assert parsed["kabupaten_kota"] == "PEMALANG"
    assert parsed["issued_place"] == "PEMALANG"
    assert parsed["issued_date"] == "13-11-2018"
    assert parsed["issued_date"] != "11-12-1989"


def test_bekasi_phone_ocr_lines():
    parsed = parse_ktp(BEKASI_KTP_LINES)

    assert parsed["nik"] == "3216010809780001"
    assert parsed["nama"] == "ANDI WIBOWO"
    assert parsed["tempat_tgl_lahir"] == "PEMALANG, 08-09-1978"
    assert parsed["jenis_kelamin"] == "LAKI-LAKI"
    assert parsed["gol_darah"] == "O"
    assert parsed["kel_desa"] == "SUKAMAJU"
    assert parsed["kecamatan"] == "CIKARANG SELATAN"
    assert parsed["status_perkawinan"] == "KAWIN"
    assert parsed["berlaku_hingga"] == "SEUMUR HIDUP"
    assert parsed["issued_place"] == "CIKARANG"
    assert parsed["issued_date"] == "04-05-2016"
    assert parsed["issued_date"] not in parsed["tempat_tgl_lahir"]


def test_bottom_issue_date_wins_when_birth_line_is_unlabeled():
    parsed = parse_ktp(
        [
            "NIK",
            "3327011112890001",
            "Nama",
            ":RISWANDI",
            "XXXX",
            "PEMALANG, 11-12-1989",
            "PEMALANG",
            "13-11-2018",
        ]
    )

    assert parsed["nama"] == "RISWANDI"
    assert parsed["issued_date"] == "13-11-2018"
    assert parsed["issued_place"] == "PEMALANG"
    assert parsed["tempat_tgl_lahir"] is None


def test_same_line_fuzzy_gender_and_blood():
    parsed = parse_ktp(
        [
            "Jens Kelamin LAKILAKI",
            "GolDarah O",
        ]
    )

    assert parsed["jenis_kelamin"] == "LAKI-LAKI"
    assert parsed["gol_darah"] == "O"


def test_empty_input_returns_null_fields():
    parsed = parse_ktp([])

    assert parsed == {key: None for key in KTP_FIELDS}


# Live service OCR text from the four previously tested cards (no images).
# Fixtures are under tests/fixtures/ktp/.

_KTP_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "ktp"


def _fixture_txts(name: str) -> list[str]:
    payload = json.loads((_KTP_FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return list(payload["txts"])


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "ebta_sleman",
            {
                "nik": "3403081205820001",
                "nama": "EBTA SETIAWAN, S.T",
                "tempat_tgl_lahir": "GUNUNGKIDUL, 12-05-1982",
                "jenis_kelamin": "LAKI-LAKI",
                "gol_darah": None,
                "alamat": "NGANGKRIK",
                "rt_rw": "001/014",
                "kel_desa": "TRIHARJO",
                "kecamatan": "SLEMAN",
                "agama": "ISLAM",
                "status_perkawinan": "KAWIN",
                "pekerjaan": "KARYAWAN SWASTA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "SEUMUR HIDUP",
                "provinsi": "DAERAH ISTIMEWA YOGYAKARTA",
                "kabupaten_kota": "SLEMAN",
                "issued_place": "SLEMAN",
                "issued_date": "14-06-2021",
            },
        ),
        (
            "garut",
            {
                "nik": "3205192301880001",
                "nama": "YOGI IRFAN ROSYADI, S.Pd.I",
                "tempat_tgl_lahir": "GARUT, 23-01-1988",
                "jenis_kelamin": "LAKI-LAKI",
                "gol_darah": None,
                "alamat": "KP. CIKANCUNG",
                "rt_rw": "001/003",
                "kel_desa": "MARGALAKSANA",
                "kecamatan": "CILAWU",
                "agama": "ISLAM",
                "status_perkawinan": "KAWIN",
                "pekerjaan": "WIRASWASTA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "SEUMUR HIDUP",
                "provinsi": "JAWA BARAT",
                "kabupaten_kota": "GARUT",
                "issued_place": "GARUT",
                "issued_date": "17-02-2020",
            },
        ),
        (
            "pemalang",
            {
                "nik": "3327101112890061",
                "nama": "RISWANDI",
                "tempat_tgl_lahir": "PEMALANG, 11-12-1989",
                "jenis_kelamin": "LAKI-LAKI",
                "gol_darah": "-",
                "alamat": "DESA PETANJUNGAN",
                "rt_rw": "017/002",
                "kel_desa": "PETANJUNGAN",
                "kecamatan": "PETARUKAN",
                "agama": "ISLAM",
                "status_perkawinan": "BELUM KAWIN",
                "pekerjaan": "WIRASWASTA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "SEUMUR HIDUP",
                "provinsi": "JAWA TENGAH",
                "kabupaten_kota": "PEMALANG",
                "issued_place": "PEMALANG",
                "issued_date": "13-11-2018",
            },
        ),
        (
            "bekasi",
            {
                "nik": "3216190809780007",
                "nama": "EKO BUDIONO BIN SUNARDI",
                "tempat_tgl_lahir": "PEMALANG, 08-09-1978",
                "jenis_kelamin": "LAKI-LAKI",
                "gol_darah": None,
                "kel_desa": "SUKARESMI",
                "kecamatan": "CIKARANG SELATAN",
                "agama": "ISLAM",
                "status_perkawinan": "KAWIN",
                "pekerjaan": "WIRASWASTA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "08-09-2017",
                "provinsi": "JAWA BARAT",
                "kabupaten_kota": "BEKASI",
                "issued_place": "BEKASI",
                "issued_date": "25-09-2012",
            },
        ),
    ],
)
def test_live_ocr_fixtures(name, expected):
    parsed = parse_ktp(_fixture_txts(name))

    for key, value in expected.items():
        assert parsed[key] == value, key


def test_blood_type_rejects_non_abo_values():
    parsed = parse_ktp(
        [
            "NIK",
            "3403081205820001",
            "Nama",
            ":BUDI SANTOSO",
            "Gol, Darah",
            "SUKAMAJU",
            "Alamat",
            "RT/RW",
            "001/001",
        ]
    )

    assert parsed["gol_darah"] is None
    assert parsed["alamat"] == "SUKAMAJU"
    assert parsed["nama"] == "BUDI SANTOSO"


def test_mangled_nama_label_does_not_become_the_name():
    parsed = parse_ktp(
        [
            "NIK",
            "3216010809780001",
            "Naa",
            "ANDI WIBOWO",
        ]
    )

    assert parsed["nama"] == "ANDI WIBOWO"


def test_ttl_stump_label_accepts_dot_separator():
    parsed = parse_ktp(
        [
            "NIK",
            "3216010809780001",
            "Nama",
            ":ANDI WIBOWO",
            "Temanr",
            "PEMALANG.08-09-1978",
        ]
    )

    assert parsed["nama"] == "ANDI WIBOWO"
    assert parsed["tempat_tgl_lahir"] == "PEMALANG, 08-09-1978"


def test_agama_and_pekerjaan_are_normalized_from_catalog():
    parsed = parse_ktp(
        [
            "Agama",
            "ISA",
            "Pekerjaan",
            "KARYAWANSWASTA",
        ]
    )

    assert parsed["agama"] == "ISLAM"
    assert parsed["pekerjaan"] == "KARYAWAN SWASTA"


def test_alamat_keeps_desa_prefix_and_corrects_trailing_place():
    parsed = parse_ktp(
        [
            "Alamat",
            ":DESA PETANLUNGAN",
            "KelDesa",
            "PETANUNGAN",
        ]
    )

    assert parsed["alamat"] == "DESA PETANJUNGAN"
    assert parsed["kel_desa"] == "PETANJUNGAN"
