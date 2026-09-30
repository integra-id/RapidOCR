# Changelog

All notable changes to this integra-id RapidOCR fork and the **rapidocr-id** service are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the rapidocr-id image (`ghcr.io/integra-id/rapidocr-id`).

`1.0.0` is the first rapidocr-id service release. It is not the upstream RapidOCR library version.

## [1.5.0] - 2026-09-30

### Added

- Optional GPU image `docker/Dockerfile.rapidocr-id.gpu`. The default image stays ONNX Runtime CPU. `RAPIDOCR_PROVIDER=cuda` requests CUDA and falls back to CPU when the provider is unavailable. `/health` and `/version` report `provider`.
- Best-effort seal and signature regions in OCR and parse JSON as `artifacts`, plus `POST /detect/artifacts`. Detection uses color, shape, and nearby words such as "stempel" or "tanda tangan". Logos can be false positives.
- `format=csv` was already present; table export remains a line-box heuristic. Markdown and HTML still use that layout.
- `docs/finetune-id.md` and `make install-id-model` describe how to fine-tune with PaddleOCR, export ONNX, and copy it into the model directory. No new weights are shipped.

## [1.4.0] - 2026-09-30

### Added

- `POST /jobs` accepts several images, PDFs, or one zip. Poll `GET /jobs/{id}` and read `GET /jobs/{id}/result`. State is stored under `RAPIDOCR_JOB_DIR` (default `/data/jobs`) and removed after `RAPIDOCR_JOB_TTL_HOURS` (default 24). An optional `webhook` URL is notified when the job finishes.
- Optional `X-API-Key` or `Authorization: Bearer` when `RAPIDOCR_API_KEY` is set. `/health` stays open. `RAPIDOCR_RATE_LIMIT` defaults to 60 requests per minute per key (or per client); `0` turns it off.
- Markdown and HTML use line boxes for headings, lists, and simple columns. `csv` exports aligned tables. This is a box heuristic, not a table-recognition model. JSON pages include a `tables` array.

## [1.3.0] - 2026-09-30

### Added

- Optional `preprocess=true|false|light` before OCR. Parse endpoints default to full phone-photo cleanup (deskew when tilted, median denoise when noisy, CLAHE). `/ocr` and PDF pages default to off. A page that is already straight skips the rotation.
- `min_score` (default `0.5`). Lines below it are marked `below_min_score` and are not used as structured field values. Parser JSON adds `field_scores`; uncertain fields are `null`.
- PDF controls on `/ocr` and `/ocr/pdf`: `pages` (`1-3` or `1,3,5`), `max_pages`, and `dpi` (72–300, default raster scale is 144 DPI). `crop=x,y,w,h` applies to an image, or only the first selected PDF page. Hard caps stay 20 pages and 20 MB.
- `POST /parse/npwp` (`npwp`, `nama`, `alamat`, `jenis_wp`) and `POST /parse/invoice` (nomor, tanggal, penjual, pembeli, NPWP, subtotal, PPN, total, currency). Both read text lines only, like KTP.

## [1.2.0] - 2026-09-30

### Added

- `POST /ocr` and `POST /ocr/pdf` accept several export formats in one OCR pass. Use comma-separated `formats=json,md,html` or repeat `format=` / `output=`.
- A multi-format response is JSON: `service`, `version`, `page_count`, `elapse`, and `outputs` with keys `json`, `markdown`, `html`, and `txt` for the requested formats.
- A single `format=` value still returns that format alone, with its previous Content-Type.

## [1.1.0] - 2026-09-30

### Added

- Multipage PDF OCR on `POST /ocr` and `POST /ocr/pdf`. Pages are rasterized with pypdfium2 (bundled pdfium; poppler is not required), at most 20 pages and 20 MB.
- Export `format` or `output` as `json` (default), `txt`, `md`/`markdown`, or `html`, with the matching Content-Type. Markdown uses RapidOCR's box layout when boxes exist.
- JSON documents include `service`, `version`, `page_count`, `elapse`, concatenated `text`, and `pages[{page, elapse, text, lines[{text, score, box}]}]`.

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

[1.5.0]: https://github.com/integra-id/RapidOCR/releases/tag/v1.5.0
[1.4.0]: https://github.com/integra-id/RapidOCR/releases/tag/v1.4.0
[1.3.0]: https://github.com/integra-id/RapidOCR/releases/tag/v1.3.0
[1.2.0]: https://github.com/integra-id/RapidOCR/releases/tag/v1.2.0
[1.1.0]: https://github.com/integra-id/RapidOCR/releases/tag/v1.1.0
[1.0.1]: https://github.com/integra-id/RapidOCR/releases/tag/v1.0.1
[1.0.0]: https://github.com/integra-id/RapidOCR/releases/tag/v1.0.0
