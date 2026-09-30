# Docker produksi OCR KTP

Image ini untuk dijalankan, bukan untuk mengembangkan mesin OCR. Model ONNX default (deteksi PP-OCRv6, klasifikasi sudut, pengenalan PP-OCRv6, bahasa `id`) diunduh saat image dibangun dan disimpan di `/opt/rapidocr/models`. Permintaan pertama tidak mengunduh apa pun. Proses memuat ketiga model saat container mulai.

Perintah dari root repositori:

```bash
docker compose -f docker/docker-compose.ktp.yml up --build
```

Hanya membangun image:

```bash
docker build -f docker/Dockerfile.ktp -t rapidocr-ktp:latest .
```

Layanan mendengarkan di port 8000.

```bash
curl -s http://127.0.0.1:8000/health
```

```bash
curl -s -F file=@ktp.jpg http://127.0.0.1:8000/ocr
curl -s -F file=@ktp.jpg http://127.0.0.1:8000/ktp
```

`POST /ocr` mengembalikan baris teks, skor, dan kotak. `POST /ktp` mengembalikan baris yang sama plus field `parse_ktp` (`nik`, `nama`, `tempat_tgl_lahir`, dan seterusnya). Unggahan adalah multipart dengan nama field `file`, paling besar 15 MB.

`GET /health` menjawab `{"status":"ok","lang":"id","models_ready":true}` setelah model siap.

Image pengembangan di `docker/docker-compose.yaml` tetap terpisah: source di-mount dan perintahnya shell.
