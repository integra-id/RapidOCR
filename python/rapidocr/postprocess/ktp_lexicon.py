# -*- encoding: utf-8 -*-
"""Common Indonesian name tokens used to restore spaces in glued KTP names.

Segmentation only replaces a token when the lexicon covers it completely.
Unknown strings are left unchanged.
"""

_NAME_TOKENS = """
abdul abdullah ade adi agung agus ahmad aisyah ajeng ali amin aminah andi
andri angga ani anita anton anwar ari arif asep ayu bagus bambang basuki bayu
binti bin budi cahya cahyadi citra dani dedi deni dewi dian diah dimas dina dwi
eka eko endang fadli fahmi farah farid fauzi fitri gita hadi hakim halim hana
hendra hendri heri heru hidayat ika imam indah indra irfan iwan joko kurnia
kurniawan lestari lia lina lisa lukman maman maya mega muhamad muhammad
mulyadi mulyana nanda ningsih nova nur nurul oki putra putri rahayu rahma
rahman rahmat rani ratna reza rian rina rio rizki rizky rohman rosyadi rudi
rusdi sari setiawan siti slamet sri suci surya susanto sutisna syamsul taufik
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
