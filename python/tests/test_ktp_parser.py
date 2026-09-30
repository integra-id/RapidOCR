# -*- encoding: utf-8 -*-
from types import SimpleNamespace

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


def test_empty_input_returns_null_fields():
    parsed = parse_ktp([])

    assert parsed == {key: None for key in KTP_FIELDS}
