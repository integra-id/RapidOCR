# Profil OCR dokumen (opt-in)

Jalur KTP dan default `/ocr` **tidak berubah**. Profil ini hanya aktif jika klien memintanya.

## Mengapa ada profil terpisah

Default layanan (`RAPIDOCR_MAX_SIDE=1280`, `RAPIDOCR_MIN_SHORT=960`, raster PDF 144 DPI, model PP-OCRv6 **small**, klasifier sudut hidup) sudah sesuai untuk foto KTP di produksi. Pengaturan itu mereduksi halaman scan A4 200 DPI dan membuat OCR dokumen lebih lambat daripada model **tiny** tanpa klasifier.

## Cara memanggil

Semua rute berikut memakai mesin dokumen, bukan mesin KTP:

```bash
# Disarankan: rute baru
curl -s -F file=@kontrak.pdf "http://127.0.0.1:8000/ocr/document?dpi=200"
curl -s -F file=@kontrak.pdf "http://127.0.0.1:8000/ocr/document/pdf"

# Setara, tetap di rute lama (hanya jika profile=document dikirim)
curl -s -F file=@kontrak.pdf "http://127.0.0.1:8000/ocr?profile=document"
curl -s -F file=@halaman1.jpg -F file=@halaman2.jpg -F task=document \
  http://127.0.0.1:8000/jobs
```

Tanpa `profile` / tanpa rute baru, perilaku sama persis seperti sekarang, termasuk `/parse/ktp`, `/parse/npwp`, `/parse/invoice`, `/ocr`, `/ocr/pdf`, dan `task=ocr`.

Parameter tambahan (hanya berlaku pada profil dokumen):

| Parameter | Bawaan | Arti |
| --- | --- | --- |
| `dpi` | 200 | Raster PDF. Jika klien mengisi `dpi`, nilai itu yang dipakai. |
| `model` | `tiny` | Divalidasi sebagai `tiny` atau `small`. Pemilihan model aktual mengikuti `RAPIDOCR_DOC_MODEL` (satu pool per proses); ganti model dengan env atau kontainer terpisah. |
| `fix_digits` | `false` | Jika `true`, huruf `o`/`O` di dalam kelompok angka diubah menjadi `0`. |
| `preprocess`, `min_score`, `pages`, `crop`, `formats` | sama seperti `/ocr` | Tidak diubah. Praproses `/ocr` tetap default `off`. |

Skema JSON respons sama dengan `/ocr` dan `/ocr/pdf` (satu gambar: `lines`; PDF: `text` + `pages`).

## Variabel lingkungan (`RAPIDOCR_DOC_*`)

Variabel lama (`RAPIDOCR_MAX_SIDE`, `RAPIDOCR_MIN_SHORT`, `RAPIDOCR_WORKERS`, …) **tidak** dipakai oleh profil dokumen dan maknanya untuk KTP tidak berubah.

| Variabel | Bawaan | Arti |
| --- | --- | --- |
| `RAPIDOCR_DOC_MODEL` | `tiny` | `tiny` atau `small`. |
| `RAPIDOCR_DOC_DPI` | `200` | DPI PDF jika klien tidak mengirim `dpi`. |
| `RAPIDOCR_DOC_MAX_SIDE` | `0` | Cap sisi panjang di layanan. `0` = tanpa downscale. |
| `RAPIDOCR_DOC_MIN_SHORT` | `0` | Lantai sisi pendek. `0` = mati. |
| `RAPIDOCR_DOC_ENGINE_MAX_SIDE` | `2000` | `Global.max_side_len` di mesin dokumen, sama dengan bawaan pustaka. Nilai 3000–4096 justru menurunkan akurasi pada pindai A4 200 DPI (PP-OCRv6 tiny + `Det.limit_side_len=736/min`). Tetap dapat dinaikkan lewat env jika dokumen lebih besar. |
| `RAPIDOCR_DOC_WORKERS` | `1` | Jumlah sesi mesin dokumen. Naikkan ke 2–4 untuk PDF banyak halaman. |
| `RAPIDOCR_DOC_INTRA_THREADS` | `cores / workers` | Thread ONNX per sesi dokumen. |
| `RAPIDOCR_DOC_PRELOAD` | kosong / mati | Jika `1`, mesin dokumen dimuat saat proses start. Jika tidak, dimuat pada permintaan dokumen pertama. |

## Memori dan Docker

Mesin default (small + klasifier) tetap seperti sekarang. Mesin dokumen (tiny, cls mati, 1 worker) dimuat malas pada permintaan dokumen pertama, atau tidak pernah jika tidak ada klien dokumen. Di proses yang sama, tambahan RSS terukur sekitar 50 MB (bobot tiny ~6 MB di disk; runtime ONNX sudah termuat oleh mesin default). Kontainer khusus profil dokumen tanpa mesin KTP kira-kira 0,4–0,5 GB, setara proses pustaka tiny saja.

Image yang sama bisa menjalankan kedua jalur. Untuk memuat model tiny saat start tanpa menyentuh layanan KTP di port 8000:

```bash
docker compose -f docker/docker-compose.rapidocr-id.yml --profile document up
```

Itu menambah kontainer di port 8001. `docker compose up` tanpa `--profile document` tetap hanya layanan produksi di port 8000.

Saat membangun image berikutnya, `python -m rapidocr.service.preload --document` bisa disertakan agar model tiny ikut ter-bake (~6 MB tambahan). Perintah preload tanpa flag itu tetap hanya mengunduh model default (small + cls), jadi image lama tidak berubah perilakunya.

## Yang tidak boleh diubah

- Default `config.yaml` (lang `id`, PP-OCRv6 small, `use_cls`, flip guard).
- Default `RAPIDOCR_MAX_SIDE=1280`, `RAPIDOCR_MIN_SHORT=960`, DPI PDF 144.
- Praproses parser KTP (default `full`).
- Isi `/health` dan `/version`.
- Versi layanan default adalah **1.5.5**.
