# -*- encoding: utf-8 -*-
"""Identity of the rapidocr-id HTTP service.

1.0.0 was the first release of this service image. 1.0.1 stabilizes the
HTTP JSON bodies. Neither number is the upstream RapidOCR library version.
Override the running version at image build time with ``RAPIDOCR_ID_VERSION``.
"""

import os

SERVICE_NAME = "rapidocr-id"
SERVICE_VERSION = os.environ.get("RAPIDOCR_ID_VERSION", "1.0.1")

# ONNX files baked into the image by ``python -m rapidocr.service.preload``.
# Sizes are the ModelScope artifacts used by the default Indonesian config.
BUNDLED_MODELS = (
    {
        "file": "PP-OCRv6_det_small.onnx",
        "role": "text detection",
        "lang": "id",
        "approx_mb": 9.5,
    },
    {
        "file": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "role": "text-line angle classification",
        "lang": "ch",
        "approx_mb": 0.6,
    },
    {
        "file": "PP-OCRv6_rec_small.onnx",
        "role": "text recognition",
        "lang": "id",
        "approx_mb": 20.3,
    },
)
