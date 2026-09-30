# -*- encoding: utf-8 -*-
"""Copy an exported ONNX file into a rapidocr-id model directory.

This does not train or convert a model. The training workflow is documented
in docs/finetune-id.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

NAMES = {
    "det": "PP-OCRv6_det_small.onnx",
    "rec": "PP-OCRv6_rec_small.onnx",
    "cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="Exported ONNX file to install")
    parser.add_argument("--role", required=True, choices=sorted(NAMES))
    parser.add_argument(
        "--dest",
        default=os.environ.get("RAPIDOCR_MODEL_DIR", "models"),
        help="Directory the service reads (RAPIDOCR_MODEL_DIR)",
    )
    args = parser.parse_args()
    source = Path(args.onnx)
    if not source.is_file():
        raise SystemExit(f"ONNX file not found: {source}")
    destination = Path(args.dest)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / NAMES[args.role]
    shutil.copy2(source, target)
    print(f"Installed {source} as {target}")
    print("Restart rapidocr-id so the new file is loaded. No model was trained.")


if __name__ == "__main__":
    main()
