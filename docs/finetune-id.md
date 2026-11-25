# Fine-tune jalur bahasa Indonesia

Dokumen ini menjelaskan cara mengganti model bawaan rapidocr-id. Sprint ini tidak melatih bobot baru. Image CPU tetap memakai PP-OCRv6 small yang sudah diunduh saat build.

## Alur

1. Kumpulkan gambar dokumen Indonesia (KTP, NPWP, faktur, surat) beserta teks ground truth per baris.
2. Fine-tune deteksi atau pengenalan dengan PaddleOCR di mesin yang terpisah dari image layanan. Ikuti panduan resmi PaddleOCR untuk `det` dan `rec`.
3. Ekspor checkpoint ke ONNX. Nama berkas yang dikenali layanan:
   - deteksi: `PP-OCRv6_det_small.onnx`
   - pengenalan: `PP-OCRv6_rec_small.onnx`
   - sudut baris: `ch_ppocr_mobile_v2.0_cls_mobile.onnx`
4. Salin berkas ONNX ke direktori model, lalu mulai ulang proses.

```bash
python python/tools/prepare_id_model.py \
  --onnx /path/exported-rec.onnx \
  --role rec \
  --dest /opt/rapidocr/models
```

Atau:

```bash
make install-id-model ARGS="--onnx /path/exported-rec.onnx --role rec --dest /opt/rapidocr/models"
```

`prepare_id_model.py` hanya menyalin berkas. Perintah itu tidak menjalankan pelatihan.

Untuk image Docker, mount direktori yang sudah berisi ONNX di `/opt/rapidocr/models`, atau bangun ulang image setelah berkas itu ada di lingkungan build. Build bawaan tetap mengunduh model publik bila berkas belum ada.

OpenVINO dan TensorRT tetap jalur pengembangan di `docker/docker-compose.yaml`. Image produksi bawaan tidak memuat runtime itu.
