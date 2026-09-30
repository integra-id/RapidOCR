# -*- encoding: utf-8 -*-
import os

# Keep the in-memory limiter from failing long test runs.
os.environ.setdefault("RAPIDOCR_RATE_LIMIT", "0")
