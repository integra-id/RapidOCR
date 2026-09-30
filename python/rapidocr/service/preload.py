# -*- encoding: utf-8 -*-
"""Download and initialize the default Indonesian OCR models."""

from pathlib import Path

from rapidocr.service.app import preload

if __name__ == "__main__":
    engine = preload()
    model_dir = Path(engine.cfg.Global.model_root_dir)
    names = sorted(path.name for path in model_dir.iterdir() if path.is_file())
    print(f"Models ready in {model_dir}: {', '.join(names)}")
