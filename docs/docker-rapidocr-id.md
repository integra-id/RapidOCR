# rapidocr-id

Layanan OCR bahasa Indonesia yang bisa dijalankan sendiri. OCR umum adalah API utama. Parser dokumen, saat ini KTP, adalah tambahan.

Versi layanan saat ini adalah **1.2.0**. `1.0.0` adalah rilis pertama. Angka itu untuk image `rapidocr-id`, bukan versi pustaka upstream RapidOCR.

## Model di dalam image

Model diunduh saat image dibangun, lalu disimpan di `/opt/rapidocr/models`. Permintaan tidak mengunduh model.

| File | Peran | Bahasa | Ukuran |
| --- | --- | --- | --- |
| `PP-OCRv6_det_small.onnx` | deteksi teks | `id` | sekitar 9,5 MB |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | sudut baris teks | bersama (`ch`) | sekitar 0,6 MB |
| `PP-OCRv6_rec_small.onnx` | pengenalan teks | `id` | sekitar 20,3 MB |

## Bangun dan jalankan

Dari root repositori:

```bash
docker compose -f docker/docker-compose.rapidocr-id.yml up --build
```

Hanya membangun image lokal `rapidocr-id:1.2.0`:

```bash
docker build -f docker/Dockerfile.rapidocr-id --build-arg RAPIDOCR_ID_VERSION=1.2.0 -t rapidocr-id:1.2.0 .
```

Tarik image yang sudah diterbitkan:

```bash
docker pull ghcr.io/integra-id/rapidocr-id:1.2.0
docker run --rm -p 8000:8000 ghcr.io/integra-id/rapidocr-id:1.2.0
```

Tag mengambang yang ikut diterbitkan: `1.0`, `1`, dan `latest`.

## API

Layanan mendengarkan di port 8000. Unggahan gambar adalah multipart dengan nama field `file`, paling besar 15 MB.

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/version
curl -s -F file=@halaman.jpg http://127.0.0.1:8000/ocr
curl -s -F file=@ktp.jpg http://127.0.0.1:8000/parse/ktp
```

Setiap respons memakai `Content-Type: application/json`. Kunci di bawah selalu ada. Nilai yang tidak diketahui adalah `null`.

`GET /health`

```json
{"status": "ok", "service": "rapidocr-id", "version": "1.2.0", "lang": "id", "models_ready": true}
```

`GET /version`

```json
{"service": "rapidocr-id", "version": "1.2.0"}
```

`POST /ocr`

```json
{
  "service": "rapidocr-id",
  "version": "1.2.0",
  "elapse": 0.42,
  "lines": [{"text": "NIK", "score": 0.99, "box": [[0, 1], [10, 1], [10, 2], [0, 2]]}]
}
```

`POST /parse/ktp` dan alias `POST /ktp`. Parser hanya memakai urutan string OCR, bukan kotak, karena kotak pada foto HP mengacak field.

```json
{
  "service": "rapidocr-id",
  "version": "1.2.0",
  "elapse": 0.42,
  "lines": [{"text": "Namá", "score": 0.97}],
  "fields": {"nik": "3327011112890001", "nama": "RISWANDI", "tempat_tgl_lahir": null}
}
```

`fields` selalu memuat seluruh kunci KTP. Contoh di atas dipotong. Kesalahan memakai `{"detail": "pesan"}`.

## PDF

`POST /ocr` dan `POST /ocr/pdf` menerima PDF multipart (`file`). Setiap halaman di-raster dengan pypdfium2 (pdfium ikut terpasang, Poppler tidak dipakai), lalu di-OCR. Batasnya 20 halaman dan 20 MB. `/parse/ktp` tetap hanya untuk gambar.

Parameter `format` atau `output` (query atau form): `json` (bawaan), `txt`, `md` atau `markdown`, `html`.

```bash
curl -s -F file=@dokumen.pdf "http://127.0.0.1:8000/ocr/pdf?format=json"
curl -s -F file=@dokumen.pdf "http://127.0.0.1:8000/ocr/pdf?format=txt"
curl -s -F file=@dokumen.pdf "http://127.0.0.1:8000/ocr?format=md"
curl -s -F file=@dokumen.pdf -F output=html http://127.0.0.1:8000/ocr/pdf
curl -s -F file=@dokumen.pdf "http://127.0.0.1:8000/ocr/pdf?formats=json,md,html"
curl -s -F file=@dokumen.pdf "http://127.0.0.1:8000/ocr?format=txt&format=json"
```

Satu nilai `format` tetap mengembalikan format itu saja. Beberapa format (`formats=json,md,html` atau `format` / `output` berulang) menjalankan OCR sekali, lalu membungkus hasilnya:

```json
{
  "service": "rapidocr-id",
  "version": "1.2.0",
  "page_count": 2,
  "elapse": 0.8,
  "outputs": {
    "json": {"service": "rapidocr-id", "version": "1.2.0", "page_count": 2, "pages": []},
    "markdown": "# Page 1\n\n...",
    "html": "<!DOCTYPE html>..."
  }
}
```

Content-Type: `application/json`, `text/plain`, `text/markdown`, atau `text/html`.

JSON PDF:

```json
{
  "service": "rapidocr-id",
  "version": "1.2.0",
  "page_count": 2,
  "elapse": 0.8,
  "text": "baris halaman satu\n\nbaris halaman dua",
  "pages": [
    {
      "page": 1,
      "elapse": 0.4,
      "text": "baris halaman satu",
      "lines": [{"text": "baris halaman satu", "score": 0.9, "box": [[1, 1], [20, 1], [20, 10], [1, 10]]}]
    }
  ]
}
```

`text` menggabungkan halaman. Markdown memakai tata letak kotak RapidOCR bila kotak ada. `txt` menandai `--- page N ---`.

## Rilis ke GHCR

Workflow `.github/workflows/publish-rapidocr-id.yml` membangun image dan mendorongnya ke `ghcr.io/integra-id/rapidocr-id`. Izin yang dipakai adalah `packages: write` pada `GITHUB_TOKEN`.

Untuk rilis `1.0.0`, naikkan versi di `python/rapidocr/service/version.py` dan `CHANGELOG.md`, commit di `development`, lalu buat tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Tag `v1.0.0` menghasilkan image `1.0.0`, `1.0`, `1`, dan `latest`.

Rilis manual tanpa tag: jalankan workflow **Publish rapidocr-id** lewat `workflow_dispatch` dan isi versi semver, misalnya `1.0.0`.

Image pengembangan di `docker/docker-compose.yaml` tetap terpisah. Compose itu me-mount source dan membuka shell.
