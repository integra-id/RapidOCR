# Changelog

All notable changes to this integra-id RapidOCR fork and the **rapidocr-id** service are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the rapidocr-id image (`ghcr.io/integra-id/rapidocr-id`).

`1.0.0` is the first rapidocr-id service release. It is not the upstream RapidOCR library version.

## [1.0.1] - 2026-09-30

### Changed

- HTTP bodies are stable JSON (`Content-Type: application/json`). `POST /ocr` returns `service`, `version`, `elapse`, and `lines` of `{text, score, box}`.
- `POST /parse/ktp` and `POST /ktp` return `service`, `version`, `elapse`, `lines` of `{text, score}`, and `fields` with nulls for missing KTP values.
- KTP parsing uses recognition strings only. Box order from phone photos is not applied, so Garut, Pemalang, and Bekasi lines stay in OCR text order.

## [1.0.0] - 2026-09-30

### Added

- Bahasa Indonesia (`id`) is the default detection and recognition language. Other languages stay selectable. The shared text-line angle model remains `ch`.
- `parse_ktp` maps OCR lines to Indonesian identity-card fields, including phone-photo label fixes and a separate issue date.
- rapidocr-id production image (`docker/Dockerfile.rapidocr-id`). ONNX Runtime CPU. These models are downloaded while the image is built and stored in `/opt/rapidocr/models`:
  - `PP-OCRv6_det_small.onnx` (detection, `id`, about 9.5 MB)
  - `ch_ppocr_mobile_v2.0_cls_mobile.onnx` (angle classification, shared, about 0.6 MB)
  - `PP-OCRv6_rec_small.onnx` (recognition, `id`, about 20.3 MB)
- HTTP API: `GET /health`, `GET /version`, `POST /ocr`, and the KTP parser at `POST /parse/ktp` (alias `POST /ktp`).
- GitHub Actions workflow publishes `vX.Y.Z` to `ghcr.io/integra-id/rapidocr-id` as `X.Y.Z`, `X.Y`, `X`, and `latest`.

[1.0.1]: https://github.com/integra-id/RapidOCR/releases/tag/v1.0.1
[1.0.0]: https://github.com/integra-id/RapidOCR/releases/tag/v1.0.0
