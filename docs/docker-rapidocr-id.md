# rapidocr-id

Layanan OCR bahasa Indonesia yang bisa dijalankan sendiri. OCR umum adalah API utama. Parser dokumen, saat ini KTP, adalah tambahan.

Versi layanan pertama adalah **1.0.0**. Angka itu untuk image `rapidocr-id`, bukan versi pustaka upstream RapidOCR.

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

Hanya membangun image lokal `rapidocr-id:1.0.0`:

```bash
docker build -f docker/Dockerfile.rapidocr-id --build-arg RAPIDOCR_ID_VERSION=1.0.0 -t rapidocr-id:1.0.0 .
```

Tarik image yang sudah diterbitkan:

```bash
docker pull ghcr.io/integra-id/rapidocr-id:1.0.0
docker run --rm -p 8000:8000 ghcr.io/integra-id/rapidocr-id:1.0.0
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

`GET /health` dan `GET /version` memuat nama layanan `rapidocr-id` dan versinya. `POST /ocr` mengembalikan baris teks, skor, dan kotak. `POST /parse/ktp` mengembalikan baris yang sama plus field KTP. `POST /ktp` adalah alias yang sama.

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
