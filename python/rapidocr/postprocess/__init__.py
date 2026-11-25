# -*- encoding: utf-8 -*-
from .invoice import INVOICE_FIELDS, parse_invoice
from .ktp import KTP_FIELDS, parse_ktp
from .npwp import NPWP_FIELDS, parse_npwp

__all__ = [
    "INVOICE_FIELDS",
    "KTP_FIELDS",
    "NPWP_FIELDS",
    "parse_invoice",
    "parse_ktp",
    "parse_npwp",
]
