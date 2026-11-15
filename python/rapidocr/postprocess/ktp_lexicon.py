# -*- encoding: utf-8 -*-
"""Common Indonesian name tokens used to restore spaces in glued KTP names.

Segmentation only replaces a token when the lexicon covers it completely.
Unknown strings are left unchanged.
"""

_NAME_TOKENS = """
abdul abdullah ade adi agung agus ahmad aisyah ajeng ali amin aminah andi
andri angga ani anita anton anwar ari arif asep ayu bagus bambang basuki bayu
binti bin budi cahya cahyadi citra dani dede dedi deni dewi dian diah dimas dina dwi
eka eko endang fadli fahmi farah farid fauzi fitri gita hadi hakim halim hana
hendra hendri heri heru hidayat ika imam indah indra irfan iwan joko kurnia
kurniawan lestari lia lina lisa lukman maman maya mega muhamad muhammad
mulyadi mulyana nanda ningsih nova nur nurul oki putra putri rahayu rahma
rahman rahmat rani ratna reza rian rina rio rizki rizky rohman rosyadi rudi
rusdi sari setiawan siti slamet sri suci supian surya susanto sutisna syamsul taufik
tri umar utami wahyu wati wawan wibowo wijaya wulan yani yanti yoga yogi yudi
yulia yunita yusuf zainal
"""

NAME_LEXICON = frozenset(
    token.strip().upper() for token in _NAME_TOKENS.split() if token.strip()
)

# Administrative words that may be glued to a place name in phone OCR.
# Longest affixes are listed first so they win over a shorter overlap.
PLACE_PREFIXES = (
    "KELURAHAN",
    "KECAMATAN",
    "KABUPATEN",
    "DESA",
)
PLACE_SUFFIXES = (
    "TENGGARA",
    "SELATAN",
    "UTARA",
    "TIMUR",
    "BARAT",
    "TENGAH",
    "PUSAT",
)

# Closed vocabularies printed on a KTP. Matching is by compact form
# (spaces, punctuation, and case removed) plus a small edit distance.
AGAMA_CANON = (
    "ISLAM",
    "KRISTEN",
    "KATOLIK",
    "HINDU",
    "BUDDHA",
    "KONGHUCU",
    "KEPERCAYAAN",
)

# Official Dukcapil "jenis pekerjaan" spellings, plus a few common aliases
# that compact to the same official phrase (PNS, TNI, POLRI).
_PEKERJAAN_OFFICIAL = """
BELUM/TIDAK BEKERJA
MENGURUS RUMAH TANGGA
PELAJAR/MAHASISWA
PENSIUNAN
PEGAWAI NEGERI SIPIL
PNS
TENTARA NASIONAL INDONESIA
TNI
KEPOLISIAN RI
POLRI
PERDAGANGAN
PETANI/PEKEBUN
PETERNAK
NELAYAN/PERIKANAN
INDUSTRI
KONSTRUKSI
TRANSPORTASI
KARYAWAN SWASTA
KARYAWAN BUMN
KARYAWAN BUMD
KARYAWAN HONORER
BURUH HARIAN LEPAS
BURUH TANI/PERKEBUNAN
BURUH NELAYAN/PERIKANAN
BURUH PETERNAKAN
PEMBANTU RUMAH TANGGA
TUKANG CUKUR
TUKANG LISTRIK
TUKANG BATU
TUKANG KAYU
TUKANG SOL SEPATU
TUKANG LAS/PANDAI BESI
TUKANG JAHIT
TUKANG GIGI
PENATA RIAS
PENATA BUSANA
PENATA RAMBUT
MEKANIK
SENIMAN
TABIB
PARAJI
PERANCANG BUSANA
PENTERJEMAH
IMAM MASJID
PENDETA
PASTOR
WARTAWAN
USTADZ/MUBALIGH
JURU MASAK
PROMOTOR ACARA
ANGGOTA DPR-RI
ANGGOTA DPD
ANGGOTA BPK
PRESIDEN
WAKIL PRESIDEN
ANGGOTA MAHKAMAH KONSTITUSI
ANGGOTA KABINET KEMENTERIAN
DUTA BESAR
GUBERNUR
WAKIL GUBERNUR
BUPATI
WAKIL BUPATI
WALIKOTA
WAKIL WALIKOTA
ANGGOTA DPRD PROVINSI
ANGGOTA DPRD KABUPATEN/KOTA
DOSEN
GURU
PILOT
PENGACARA
NOTARIS
ARSITEK
AKUNTAN
KONSULTAN
DOKTER
BIDAN
PERAWAT
APOTEKER
PSIKIATER/PSIKOLOG
PENYIAR TELEVISI
PENYIAR RADIO
PELAUT
PENELITI
SOPIR
PIALANG
PARANORMAL
PEDAGANG
PERANGKAT DESA
KEPALA DESA
BIARAWATI
WIRASWASTA
LAINNYA
"""

PEKERJAAN_CANON = tuple(
    line.strip() for line in _PEKERJAAN_OFFICIAL.splitlines() if line.strip()
)

# Alias compact forms that should display as a longer official phrase.
PEKERJAAN_ALIAS = {
    "PNS": "PEGAWAI NEGERI SIPIL",
    "TNI": "TENTARA NASIONAL INDONESIA",
    "POLRI": "KEPOLISIAN RI",
}

# A, B, AB, O, or a printed dash when the blood type is blank.
GOL_DARAH_CANON = ("A", "B", "AB", "O", "-")

# Offline kel/desa and kecamatan word list used to correct short OCR drops.
# Extend this list as more cards are added; do not special-case one card.
_PLACE_NAMES = """
TRIHARJO
NGANGKRIK
SLEMAN
GUNUNGKIDUL
PETANJUNGAN
PETARUKAN
PEMALANG
ULUJAMI
SUKARESMI
CIKARANG
CIKARANG SELATAN
BEKASI
MARGALAKSANA
CILAWU
GARUT
SUKAMAJU
GAMBIR
CIKANCUNG
RENGED
KRESEK
TEGAL KAMAL
JATINEGARA
CAKUNG
PONDOK KOPI
GROGOL SELATAN
KEBAYORAN LAMA
KEBAYORAN BARU
KEBAYORAN
JAKARTA TIMUR
JAKARTA SELATAN
JAKARTA BARAT
JAKARTA UTARA
JAKARTA PUSAT
RANDUWATANG
RANDUWATANG KIDUL
KUDU
JOMBANG
MAMUJU
TANGERANG
BANTEN
CIDODOL
"""

PLACE_GAZETTEER = tuple(
    line.strip().upper() for line in _PLACE_NAMES.splitlines() if line.strip()
)

STATUS_KAWIN_CANON = (
    "BELUM KAWIN",
    "KAWIN",
    "CERAI HIDUP",
    "CERAI MATI",
)

# Kota/kabupaten lines sometimes omit the KABUPATEN/KOTA prefix, especially
# on DKI cards ("JAKARTA TIMUR" under "PROVINSI DKI JAKARTA").
KABUPATEN_KOTA_CANON = (
    "JAKARTA TIMUR",
    "JAKARTA SELATAN",
    "JAKARTA BARAT",
    "JAKARTA UTARA",
    "JAKARTA PUSAT",
    "TANGERANG",
    "KOTA TANGERANG",
    "KABUPATEN TANGERANG",
    "JOMBANG",
    "MAMUJU",
    "SLEMAN",
    "GARUT",
    "PEMALANG",
    "BEKASI",
    "KOTA BEKASI",
    "KABUPATEN BEKASI",
    "GUNUNGKIDUL",
)

# Compact typo repairs for cardinal directions in place names.
DIRECTION_TYPOS = {
    "TUMUR": "TIMUR",
    "TMUR": "TIMUR",
    "SELTAN": "SELATAN",
    "SLATAN": "SELATAN",
    "UTATA": "UTARA",
    "BARAI": "BARAT",
}
