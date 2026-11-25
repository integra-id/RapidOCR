# -*- encoding: utf-8 -*-
"""Best-effort seal and signature regions.

Red or blue compact blobs are treated as stamps. Wide, sparse dark strokes
are treated as signatures. Nearby OCR words such as "stempel" or "tanda
tangan" add a low-confidence cue. Logos and printed headings can look like
either, so scores are hints, not proof.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import cv2
import numpy as np

_SEAL_WORDS = re.compile(r"stempel|cap\s+dinas|meterai", re.I)
_SIGN_WORDS = re.compile(r"tanda\s*tangan|\bttd\b|hormat\s+kami", re.I)


def detect_artifacts(
    image: np.ndarray, lines: Optional[list[dict[str, Any]]] = None
) -> list[dict]:
    if image is None or image.size == 0:
        return []
    found = _color_seals(image)
    found.extend(_signature_strokes(image))
    found.extend(_text_cues(lines or []))
    return _dedupe(found)


def _color_seals(image: np.ndarray) -> list[dict]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255)) | cv2.inRange(
        hsv, (168, 70, 50), (179, 255, 255)
    )
    blue = cv2.inRange(hsv, (95, 60, 40), (135, 255, 255))
    mask = cv2.morphologyEx(red | blue, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    height, width = image.shape[:2]
    page = float(height * width)
    artifacts = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < page * 0.002 or area > page * 0.2:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        _x, _y, box_w, box_h = cv2.boundingRect(contour)
        aspect = box_w / max(box_h, 1)
        if circularity < 0.45 and not 0.7 <= aspect <= 1.4:
            continue
        score = float(max(0.35, min(0.95, circularity)))
        artifacts.append(_artifact("seal", contour, score))
    return artifacts


def _signature_strokes(image: np.ndarray) -> list[dict]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    # Stamps are handled separately. Ignore strongly colored pixels.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colorful = hsv[:, :, 1] > 80
    ink[colorful] = 0
    roi = ink.copy()
    roi[: int(height * 0.35), :] = 0
    contours, _hierarchy = cv2.findContours(
        roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    artifacts = []
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        if box_h < 8 or box_w < width * 0.18:
            continue
        if box_w / max(box_h, 1) < 2.2:
            continue
        if box_h > height * 0.4:
            continue
        region = ink[y : y + box_h, x : x + box_w]
        density = float(np.count_nonzero(region)) / float(region.size)
        if not 0.02 <= density <= 0.4:
            continue
        score = float(max(0.3, min(0.85, 1.0 - density)))
        artifacts.append(_artifact("signature", contour, score))
    return artifacts


def _text_cues(lines: list[dict[str, Any]]) -> list[dict]:
    artifacts = []
    for line in lines:
        text = line.get("text") or ""
        box = line.get("box")
        if box is None:
            continue
        if _SEAL_WORDS.search(text):
            kind = "seal"
        elif _SIGN_WORDS.search(text):
            kind = "signature"
        else:
            continue
        artifacts.append(
            {
                "type": kind,
                "box": [[float(point[0]), float(point[1])] for point in box],
                "score": 0.4,
            }
        )
    return artifacts


def _artifact(kind: str, contour, score: float) -> dict:
    x, y, box_w, box_h = cv2.boundingRect(contour)
    box = [
        [float(x), float(y)],
        [float(x + box_w), float(y)],
        [float(x + box_w), float(y + box_h)],
        [float(x), float(y + box_h)],
    ]
    return {"type": kind, "box": box, "score": round(score, 3)}


def _dedupe(artifacts: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for artifact in sorted(artifacts, key=lambda item: item["score"], reverse=True):
        if any(_iou(artifact["box"], previous["box"]) > 0.45 for previous in kept):
            continue
        kept.append(artifact)
    return kept


def _iou(left, right) -> float:
    ax1, ay1 = left[0]
    ax2, ay2 = left[2]
    bx1, by1 = right[0]
    bx2, by2 = right[2]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_left = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_right = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_left + area_right - intersection
    if union <= 0:
        return 0.0
    return intersection / union
