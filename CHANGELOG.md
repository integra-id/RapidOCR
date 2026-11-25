# Changelog

All notable changes to this integra-id RapidOCR fork and the **rapidocr-id** service are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the rapidocr-id image (`ghcr.io/integra-id/rapidocr-id`).

`1.0.0` is the first rapidocr-id service release. It is not the upstream RapidOCR library version.

## [Unreleased]

### Added

- Opt-in **document OCR profile** for full-page scans (DMS and similar) without changing the production KTP path. New routes: `POST /ocr/document`, `POST /ocr/document/pdf`, and `POST /jobs` with `task=document`. Existing `POST /ocr` and `POST /ocr/pdf` also accept `profile=document`. Clients that omit the new parameter keep the current defaults, models, preprocessing, downscale, PDF DPI, and JSON schema.
- Document profile defaults: PDF raster at 200 DPI, no `RAPIDOCR_MAX_SIDE` / `RAPIDOCR_MIN_SHORT` downscale, `Global.max_side_len=2000` (library default; 3000+ dropped character accuracy on 200 DPI A4 scans), text-line classifier off, PP-OCRv6 **tiny** (override with `model=small` or `RAPIDOCR_DOC_MODEL`). Settings use the `RAPIDOCR_DOC_*` prefix. The document engine is loaded lazily on the first document request (`RAPIDOCR_DOC_PRELOAD=1` to load at startup). Optional `fix_digits=true` applies an `o`/`O` → `0` repair inside digit groups; it is off by default.
- Optional Compose service `rapidocr-id-document` behind the `document` profile, same image, separate port. Default `docker compose up` is unchanged.

## [1.5.4] - 2026-11-23

### Fixed

- When the text-line angle classifier rotates a crop 180°, recognition is also run on the unflipped crop and the higher-scoring text is kept (`Global.cls_score_flip_guard`, default on). A confident but wrong flip on a blurry OCR-A NIK line no longer turns the digits into garbage and drops the field.
- Full preprocess no longer median-blurs sharp KTP scans whose security-pattern texture only barely exceeded the old residual threshold of 6. The denoise residual floor is now 8.0, which still blurs grainier shots.
- The KTP parser reads a Gol. Darah value that OCR split onto its own box, peels garbled `M.B.A.` fragments such as `S.T,M B.A.` as `MBA` (same style as `S.T., MBA`), inserts a space before a glued Roman numeral on a street (`TONGKOLIII/43`), recovers a street line when an empty Gol. Darah sits above RT/RW, and strips OCR junk around `WNI` (`Kewarganegaraan_: WNI`).

### Changed

- A twelfth KTP regression card (DIY Sleman) is covered by a text-only fixture and the default `/parse/ktp` path. The image is not committed.

## [1.5.3] - 2026-11-18

### Added

- KTP parser peels academic and honorific titles that are glued to or written next to the name (`S.T.`, `S.Pd.I`, `MBA`, `Dr.`, `Hj.`, and similar) and formats `nama` as `NAME, S.T., MBA`. Optional `gelar` and `nama_tanpa_gelar` are returned alongside the full `nama`.
- `tempat_lahir` and `tgl_lahir` are split out of `tempat_tgl_lahir`. A birth date on the next line is joined in, dates are validated, and a missing or invalid date (and a missing gender) can be filled from NIK digits 7-12. A valid printed date is kept even when it disagrees with the NIK.
- Offline `ktp_wilayah.json` maps NIK digits 1-4 to the 38 provinces and 514 kabupaten/kota, and fuzzy-matches chewed headers such as `PROVINSIDAEBAHISTIMEWAYO`. Short four-letter typos like `SVAN` are not forced into a nearby name.

### Changed

- `gol_darah` now reads values after punctuation (`Gol Darah .O`, `:O`) and still maps a lone `0` next to the label to `O`. Empty Gol. Darah still defaults to `-`.
- Broken labels (`Alarnal`, `Ke/Desa`, `kerjaan`, `vganegaraan`, `Hingga`, `Jonis L.elamin`, `TgiLahir`) are matched more loosely. Unlabeled `WNI` / `SEUMUR HIDUP` can be inferred when the slot is still empty.
- Village names around Sleman (`AMBARKETAWANG`, `GAMPING`, and neighbours) were added to the place gazetteer.
- `/parse/ktp` now also returns `nama_tanpa_gelar`, `gelar`, `tempat_lahir`, and `tgl_lahir`.
- `RAPIDOCR_MAX_SIDE` stays 1280, but `RAPIDOCR_MIN_SHORT` (default 960) stops a skinny photo from being crushed below that short side. The worn Sleman KTP (2304×4096) is therefore sent at 960×1707 so Gol. Darah and Kecamatan GAMPING still read, while a 3072×4096 photo still caps at 1280. Set `RAPIDOCR_MIN_SHORT=0` for a hard long-side cap. The engine's `Global.max_side_len` is raised to 8192 when the floor is on, so the service resize is not undone. `/health` and `/version` report `min_short`.

## [1.5.2] - 2026-11-15

### Added

- Empty `gol_darah` is filled with `-` when the Gol. Darah label is present but has no valid A/B/AB/O (optional `+/-`) value, or when a NIK and gender were already parsed. The printed dash is often dropped by OCR.
- `RAPIDOCR_MAX_SIDE` (default 1280) caps the longest image side before OCR so a 3072×4096 phone photo stays near 0.6s on CPU. Set `0` to keep the library cap of 2000. The same cap is passed as `Global.max_side_len` on each engine.
- `RAPIDOCR_WORKERS` runs a pool of in-process OCR sessions (default: half the CPU count, clamped to 2–4). The global inference lock is gone, and HTTP handlers run inference in a thread pool so the event loop is not blocked. Concurrent HTTP and `/jobs` pages can run at the same time. `RAPIDOCR_INTRA_THREADS` defaults to `cores / workers` so the sessions share the machine fairly. Prefer this pool over `uvicorn --workers`; multiple OS processes would reload the models.

### Changed

- `/health` and `/version` report `workers` and `max_side`.

## [1.5.1] - 2026-10-11

### Changed

- Merged upstream RapidAI/RapidOCR: YAML-based model routing (`default_models.yaml`), det `limit_side_len` fix, rec `model_root_dir` fix, loading images from raw memory buffers, and model SHA256 fix.
- Indonesian `id` routes (with aliases such as `ind`, `indonesian`) now live in `default_models.yaml` and keep the PP-OCRv6 small det/rec models.
- `opencv-python-headless` is pinned explicitly in `requirements-service.txt`, since upstream made OpenCV optional and the service needs `cv2` for preprocess and artifacts.

### Breaking

- Unknown language names now raise `ValueError` instead of being matched loosely.

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
