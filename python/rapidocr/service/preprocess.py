# -*- encoding: utf-8 -*-
"""Light phone-photo cleanup before OCR.

Steps, in order, when ``mode`` is ``full``:

- Deskew: estimate the text tilt and rotate only when it is about 0.8 to 15
  degrees. A straight page is left unchanged.
- Denoise: a 3x3 median blur, only when a cheap noise residual is high.
- Contrast: CLAHE on the lightness channel so faint phone photos stay readable.

``light`` runs CLAHE only. ``off`` returns the image unchanged.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from rapidocr.service.pdf import DocumentError


def preprocess_bgr(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "off":
        return image
    if mode == "light":
        return _clahe(image)
    if mode != "full":
        raise DocumentError("preprocess must be true, false, or light.")
    corrected = _deskew(image)
    corrected = _maybe_denoise(corrected)
    return _clahe(corrected)


def parse_preprocess(value: Optional[str], default: str) -> str:
    if value is None or str(value).strip() == "":
        return default
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on", "full"}:
        return "full"
    if token in {"0", "false", "no", "off"}:
        return "off"
    if token == "light":
        return "light"
    raise DocumentError("preprocess must be true, false, or light.")


def apply_crop(image: np.ndarray, spec: Optional[str]) -> np.ndarray:
    """Crop ``x,y,w,h`` in pixels. Values are clamped to the image."""
    if spec is None or str(spec).strip() == "":
        return image
    parts = [part.strip() for part in str(spec).split(",")]
    if len(parts) != 4:
        raise DocumentError("crop must be x,y,w,h in pixels.")
    try:
        x, y, width, height = [int(float(part)) for part in parts]
    except ValueError as exc:
        raise DocumentError("crop must be x,y,w,h in pixels.") from exc
    if width <= 0 or height <= 0:
        raise DocumentError("crop width and height must be positive.")

    image_height, image_width = image.shape[:2]
    x = min(max(x, 0), image_width - 1)
    y = min(max(y, 0), image_height - 1)
    x2 = min(image_width, x + width)
    y2 = min(image_height, y + height)
    cropped = image[y:y2, x:x2]
    if cropped.size == 0:
        raise DocumentError("crop is outside the image.")
    return cropped


def _deskew(image: np.ndarray) -> np.ndarray:
    angle = _skew_degrees(image)
    if angle is None or abs(angle) < 0.8 or abs(angle) > 15:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _skew_degrees(image: np.ndarray) -> Optional[float]:
    height, width = image.shape[:2]
    if min(height, width) < 80:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _threshold, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    points = cv2.findNonZero(binary)
    if points is None or len(points) < 80:
        return None
    angle = cv2.minAreaRect(points)[-1]
    if angle < -45:
        angle = 90 + angle
    else:
        angle = angle
    # minAreaRect reports the box orientation. Convert to a small correction
    # that rotates the long edge toward horizontal.
    if angle > 45:
        angle -= 90
    return float(angle)


def _maybe_denoise(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    residual = cv2.absdiff(gray, cv2.medianBlur(gray, 3))
    if float(residual.mean()) < 6.0:
        return image
    return cv2.medianBlur(image, 3)


def _clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, green, red = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    merged = cv2.merge((clahe.apply(lightness), green, red))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
