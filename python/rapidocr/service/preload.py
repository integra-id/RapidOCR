# -*- encoding: utf-8 -*-
"""Download and initialize the default Indonesian OCR models."""

import argparse
from pathlib import Path

from rapidocr.service.app import preload
from rapidocr.service.document import document_preload_enabled, preload_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--document",
        action="store_true",
        help="Also download and initialize document-profile models (PP-OCRv6 tiny).",
    )
    args = parser.parse_args()

    engine = preload()
    model_dir = Path(engine.cfg.Global.model_root_dir)
    if args.document or document_preload_enabled():
        preload_document()
    names = sorted(path.name for path in model_dir.iterdir() if path.is_file())
    print(f"Models ready in {model_dir}: {', '.join(names)}")


if __name__ == "__main__":
    main()
