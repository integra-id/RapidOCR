# Parser NPWP

`parse_npwp` membaca baris OCR menjadi kartu NPWP. Endpoint HTTP-nya `POST /parse/npwp`. Parser memakai urutan teks, bukan urutan kotak.

```bash
curl -s -F file=@npwp.jpg http://127.0.0.1:8000/parse/npwp
```

Foto ponsel memakai praproses penuh secara bawaan (`preprocess=true`). Matikan dengan `preprocess=false`. `min_score` bawaan `0.5`: baris di bawahnya tidak dipakai sebagai nilai field.

Field, atau `null` bila tidak ada:

`npwp`, `nama`, `alamat`, `jenis_wp`.

Nomor NPWP dikembalikan sebagai 15 atau 16 digit. Respons juga memuat `field_scores` untuk tiap field.
