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
    assert parsed["jenis_kelamin"] == "LAKI-LAKI"
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
    assert parsed["tempat_tgl_lahir"] == "PEMALANG, 11-12-1989"
    assert parsed["tgl_lahir"] == "11-12-1989"


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


# Live service OCR text from previously tested cards (no images).
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
                "nama": "EBTA SETIAWAN, S.T.",
                "tempat_tgl_lahir": "GUNUNGKIDUL, 12-05-1982",
                "jenis_kelamin": "LAKI-LAKI",
                "gol_darah": "-",
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
                "gol_darah": "-",
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
                "gol_darah": "-",
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
        (
            "tangerang",
            {
                "nik": "3603060105040002",
                "nama": "DEDE SUPIAN",
                "tempat_tgl_lahir": "TANGERANG, 01-04-2004",
                "jenis_kelamin": "LAKI-LAKI",
                "gol_darah": "-",
                "alamat": "KP. TEGAL KAMAL",
                "rt_rw": "008/003",
                "kel_desa": "RENGED",
                "kecamatan": "KRESEK",
                "agama": "ISLAM",
                "status_perkawinan": "BELUM KAWIN",
                "pekerjaan": "BELUM/TIDAK BEKERJA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "SEUMUR HIDUP",
                "provinsi": "BANTEN",
                "kabupaten_kota": "TANGERANG",
                "issued_place": "TANGERANG",
                "issued_date": "10-04-2023",
            },
        ),
        (
            "jakarta_timur",
            {
                "nik": "3216231006900004",
                "nama": "ERIP SARIPUDIN",
                "tempat_tgl_lahir": "BEKASI, 10-05-1990",
                "jenis_kelamin": "LAKI-LAKI",
                "gol_darah": "O",
                "alamat": "PONDOK KOPI",
                "rt_rw": "001/001",
                "kel_desa": "JATINEGARA",
                "kecamatan": "CAKUNG",
                "agama": "ISLAM",
                "status_perkawinan": "KAWIN",
                "pekerjaan": "WIRASWASTA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "SEUMUR HIDUP",
                "provinsi": "DKI JAKARTA",
                "kabupaten_kota": "JAKARTA TIMUR",
                "issued_date": "02-08-2023",
            },
        ),
        (
            "jakarta_selatan",
            {
                "nik": "3174050601941002",
                "nama": "WAHYUDI",
                "tempat_tgl_lahir": "JAKARTA, 06-01-1994",
                "jenis_kelamin": "LAKI-LAKI",
                "gol_darah": "-",
                "alamat": "JL. MASJID CIDODOL NO 92",
                "rt_rw": "003/011",
                "kel_desa": "GROGOL SELATAN",
                "kecamatan": "KEBAYORAN LAMA",
                "agama": "ISLAM",
                "status_perkawinan": "KAWIN",
                "pekerjaan": "KARYAWAN SWASTA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "SEUMUR HIDUP",
                "provinsi": "DKI JAKARTA",
                "kabupaten_kota": "JAKARTA SELATAN",
                "issued_date": "24-06-2017",
            },
        ),
        (
            "jombang",
            {
                "nik": "3517171001910001",
                "nama": "SOFYAN AINUR ROZIQI",
                "tempat_tgl_lahir": "JOMBANG, 10-01-1991",
                "jenis_kelamin": "LAKI-LAKI",
                "gol_darah": "-",
                "alamat": "DSN. RANDUWATANG KIDUL",
                "rt_rw": "003/002",
                "kel_desa": "RANDUWATANG",
                "kecamatan": "KUDU",
                "agama": "ISLAM",
                "status_perkawinan": "BELUM KAWIN",
                "pekerjaan": "PELAJAR/MAHASISWA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "10-01-2017",
                "provinsi": "JAWA TIMUR",
                "kabupaten_kota": "JOMBANG",
            },
        ),
        (
            "mamuju_a",
            {
                "nik": "7600001234567890",
                "nama": "NAMA LENGKAP",
                "tempat_tgl_lahir": "MAMUJU, 01-01-1990",
                "jenis_kelamin": "LAKI-LAKI",
                "gol_darah": "O",
                "alamat": "JL. CONTOH ALAMAT",
                "rt_rw": "001/002",
                "kel_desa": "KELURAHAN CONTOH",
                "kecamatan": "KECAMATAN CONTOH",
                "agama": "ISLAM",
                "status_perkawinan": "KAWIN",
                "pekerjaan": "WIRASWASTA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "SEUMUR HIDUP",
                "provinsi": "SULAWESI BARAT",
                "kabupaten_kota": "MAMUJU",
                "issued_place": "MAMUJU",
                "issued_date": "01-06-2024",
            },
        ),
        (
            "mamuju_b",
            {
                "nik": "7600001234567890",
                "nama": "NAMA LENGKAP",
                "tempat_tgl_lahir": "MAMUJU, 01-01-1990",
                "jenis_kelamin": "LAKI-LAKI",
                "gol_darah": "O",
                "rt_rw": "001/002",
                "kel_desa": "KELURAHAN CONTOH",
                "agama": "ISLAM",
                "status_perkawinan": "KAWIN",
                "pekerjaan": "WIRASWASTA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "SEUMUR HIDUP",
                "provinsi": "SULAWESI BARAT",
                "kabupaten_kota": "MAMUJU",
                "issued_place": "MAMUJU",
                "issued_date": "01-01-2017",
            },
        ),
        (
            "endar_sleman",
            {
                "nik": "3404112805790001",
                "nama": "ENDAR KUSMARWANTO, S.T., MBA",
                "nama_tanpa_gelar": "ENDAR KUSMARWANTO",
                "gelar": "S.T., MBA",
                "tempat_tgl_lahir": "SLEMAN, 28-05-1979",
                "tempat_lahir": "SLEMAN",
                "tgl_lahir": "28-05-1979",
                "jenis_kelamin": "LAKI-LAKI",
                "gol_darah": "O",
                "alamat": "KALIMANJUNG",
                "rt_rw": "005/032",
                "kel_desa": "AMBARKETAWANG",
                "kecamatan": "GAMPING",
                "agama": "ISLAM",
                "status_perkawinan": "KAWIN",
                "pekerjaan": "KARYAWAN SWASTA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "SEUMUR HIDUP",
                "provinsi": "DAERAH ISTIMEWA YOGYAKARTA",
                "kabupaten_kota": "SLEMAN",
                "issued_place": "SLEMAN",
                "issued_date": "15-10-2016",
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

    assert parsed["gol_darah"] == "-"
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



def test_labeled_template_nik_is_kept():
    parsed = parse_ktp(["NIK", "7600001234567890", "Nama", "NAMA LENGKAP"])

    assert is_valid_nik("7600001234567890") is False
    assert parsed["nik"] == "7600001234567890"
    assert parsed["nama"] == "NAMA LENGKAP"


def test_dotted_birth_date_and_gender_typo():
    parsed = parse_ktp(
        [
            "Tempal/Tgi Lahir",
            "JOMBANG.10.01-1991",
            "Jenis Ketanin",
            "LAKI LANI",
            "Status Perluwinan: BEL UM KAWIN",
            "Berlaku Hingga",
            ":SEUMURHIDUP",
        ]
    )

    assert parsed["tempat_tgl_lahir"] == "JOMBANG, 10-01-1991"
    assert parsed["jenis_kelamin"] == "LAKI-LAKI"
    assert parsed["status_perkawinan"] == "BELUM KAWIN"
    assert parsed["berlaku_hingga"] == "SEUMUR HIDUP"


def test_dki_kota_header_without_kota_prefix():
    parsed = parse_ktp(
        [
            "PROVINSI DKI JAKARTA",
            "JAKARTA TIMUR",
            "NIK",
            "3216231006900004",
        ]
    )

    assert parsed["provinsi"] == "DKI JAKARTA"
    assert parsed["kabupaten_kota"] == "JAKARTA TIMUR"


def test_empty_gol_darah_label_defaults_to_dash():
    parsed = parse_ktp(
        [
            "NIK : 3403081205820001",
            "Nama : BUDI SANTOSO",
            "Jenis Kelamin : LAKI-LAKI",
            "Gol. Darah :",
            "Alamat : NGANGKRIK",
        ]
    )
    assert parsed["gol_darah"] == "-"
    assert parsed["alamat"] == "NGANGKRIK"
    assert parsed["jenis_kelamin"] == "LAKI-LAKI"


def test_blood_type_keeps_optional_rh():
    parsed = parse_ktp(
        [
            "NIK : 3403081205820001",
            "Nama : BUDI SANTOSO",
            "Jenis Kelamin : PEREMPUAN",
            "Gol. Darah : O+",
        ]
    )
    assert parsed["gol_darah"] == "O+"


def test_empty_input_still_leaves_gol_darah_null():
    parsed = parse_ktp([])
    assert parsed["gol_darah"] is None

def test_glued_titles_and_dot_blood_and_next_line_birth_date():
    parsed = parse_ktp(
        [
            "NIK 3404112805790001",
            "Nama :ENDAR KUSMARWANTOS.T.,MBA",
            "Tempat/TgiLahir: SLEMAN",
            "28-05-1979",
            "Gol Darah .O",
        ]
    )

    assert parsed["nama"] == "ENDAR KUSMARWANTO, S.T., MBA"
    assert parsed["nama_tanpa_gelar"] == "ENDAR KUSMARWANTO"
    assert parsed["gelar"] == "S.T., MBA"
    assert parsed["tempat_tgl_lahir"] == "SLEMAN, 28-05-1979"
    assert parsed["tempat_lahir"] == "SLEMAN"
    assert parsed["tgl_lahir"] == "28-05-1979"
    assert parsed["gol_darah"] == "O"


def test_existing_titles_stay_canonical():
    assert parse_ktp(["Nama :EBTA SETIAWAN, S.T"])["nama"] == "EBTA SETIAWAN, S.T."
    assert parse_ktp(["Nama :YOGI IRFAN ROSYADI, S.Pd.I"])["nama"] == "YOGI IRFAN ROSYADI, S.Pd.I"
    parsed = parse_ktp(["Nama :Dr. BUDI SANTOSO, M.M."])
    assert parsed["nama"] == "Dr. BUDI SANTOSO, M.M."
    assert parsed["gelar"] == "Dr., M.M."
    assert parsed["nama_tanpa_gelar"] == "BUDI SANTOSO"


def test_nik_fills_missing_birth_date_without_overwriting_a_valid_one():
    filled = parse_ktp(
        [
            "NIK 3404112805790001",
            "Nama :ENDAR KUSMARWANTO",
            "Tempat/Tgl Lahir : SLEMAN",
        ]
    )
    assert filled["tgl_lahir"] == "28-05-1979"
    assert filled["tempat_tgl_lahir"] == "SLEMAN, 28-05-1979"
    assert filled["jenis_kelamin"] == "LAKI-LAKI"

    kept = parse_ktp(
        [
            "NIK 3216231006900004",
            "Nama :ERIP SARIPUDIN",
            "Tempat/Tgl Lahir : BEKASI, 10-05-1990",
        ]
    )
    # Printed date stays when it is valid, even if NIK digits 7-12 differ.
    assert kept["tgl_lahir"] == "10-05-1990"


def test_wilayah_typos_and_short_names():
    from rapidocr.postprocess import ktp_wilayah

    assert ktp_wilayah.match_region("DAEBAHISTIMEWAYO", ktp_wilayah.province_names()) == (
        "DAERAH ISTIMEWA YOGYAKARTA"
    )
    assert ktp_wilayah.match_region("SVAN", ktp_wilayah.regency_names()) is None
    assert ktp_wilayah.nik_region("3404112805790001") == (
        "DAERAH ISTIMEWA YOGYAKARTA",
        "SLEMAN",
    )


def test_endar_column_interleaved_ocr():
    """Worn-card reading order parks the birth place between Nama and the name."""
    parsed = parse_ktp(
        [
            "PROVINSIDAERAHISTIMEWAYO",
            "KABUPATENSVAN",
            "NIK",
            "3404112805790001",
            "Tempat/TglLabir",
            "Nama",
            "SLEMAN",
            ":ENDAR KUSMARWANTOS.T.,MBA",
            "28-05-1979",
            "Janis kelamin",
            "LAKI-LAKI",
            "Alarnat",
            "KALIMANJUNG",
            "KelDesa AMBARETAWANG",
            "RT/RW",
            "005/032",
            "Kecamatan",
            "Agama",
            "ISLAM",
            "Status Perkawinan: KAWiN",
            "SLEMAN",
            "eaan",
            "KARYAWAN SWASTA",
            "15-10-2016",
            "vganegaraan:WNI",
            "Hingga",
            "SEUMURHIDUP",
        ]
    )

    assert parsed["nama"] == "ENDAR KUSMARWANTO, S.T., MBA"
    assert parsed["gelar"] == "S.T., MBA"
    assert parsed["tempat_tgl_lahir"] == "SLEMAN, 28-05-1979"
    assert parsed["alamat"] == "KALIMANJUNG"
    assert parsed["kel_desa"] == "AMBARKETAWANG"
    assert parsed["kewarganegaraan"] == "WNI"
    assert parsed["berlaku_hingga"] == "SEUMUR HIDUP"
    assert parsed["provinsi"] == "DAERAH ISTIMEWA YOGYAKARTA"
    assert parsed["kabupaten_kota"] == "SLEMAN"
    assert parsed["issued_place"] == "SLEMAN"
    assert parsed["issued_date"] == "15-10-2016"
    # Gol. Darah and Kecamatan were not present in this OCR pass.
    assert parsed["gol_darah"] == "-"
    assert parsed["kecamatan"] is None


def test_glued_kel_desa_prefix_is_peeled():
    parsed = parse_ktp(["KeVDesaAMBARKETAWANG", "Kecamatan :GAMPING"])

    assert parsed["kel_desa"] == "AMBARKETAWANG"
    assert parsed["kecamatan"] == "GAMPING"
    # Spaced sample-card values must not be peeled as a glued label.
    assert parse_ktp(["Nama", "NAMA LENGKAP"])["nama"] == "NAMA LENGKAP"
