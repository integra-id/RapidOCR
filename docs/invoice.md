# Parser faktur

`parse_invoice` membaca baris OCR faktur atau faktur pajak Indonesia. Endpoint HTTP-nya `POST /parse/invoice`. Label seperti `Faktur Pajak`, `Nomor Faktur`, `Invoice No`, `DPP`, dan `PPN` dicocokkan secara longgar.

```bash
curl -s -F file=@faktur.jpg "http://127.0.0.1:8000/parse/invoice?min_score=0.5"
```

Field, atau `null` bila tidak ada:

`nomor_invoice`, `tanggal`, `nama_penjual`, `nama_pembeli`, `npwp_penjual`, `npwp_pembeli`, `subtotal`, `ppn`, `total`, `currency`.

Jumlah uang dinormalisasi (`1.000.000` menjadi `1000000`). `Rp` atau `IDR` mengisi `currency` dengan `IDR`. Praproses foto dan ambang skor sama dengan parser NPWP dan KTP.
