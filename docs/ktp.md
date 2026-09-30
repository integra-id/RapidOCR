# Parser KTP

`parse_ktp` membaca baris OCR (dan kotak baris opsional) menjadi field KTP. Ini bukan model baru: deteksi dan pengenalan tetap RapidOCR. Bahasa lain tidak diubah.

```python
from rapidocr import RapidOCR, parse_ktp

engine = RapidOCR()  # default bahasa: id
result = engine("ktp.jpg")
fields = parse_ktp(result.txts, boxes=result.boxes)
```

`result` dari RapidOCR juga bisa diberikan langsung: `parse_ktp(result)`.

Field yang dikembalikan, atau `None` bila tidak ada:

`nik`, `nama`, `tempat_tgl_lahir`, `jenis_kelamin`, `gol_darah`, `alamat`, `rt_rw`, `kel_desa`, `kecamatan`, `agama`, `status_perkawinan`, `pekerjaan`, `kewarganegaraan`, `berlaku_hingga`, `provinsi`, `kabupaten_kota`, `issued_place`, `issued_date`.

Yang dibersihkan dari hasil OCR:

- NIK harus 16 digit dengan tanggal lahir yang masuk akal (hari perempuan ditambah 40).
- Titik dua di awal nilai dibuang.
- Header yang menempel dipisah, misalnya `PROVINSIJAWA BARAT` dan `KABUPATENGARUT`.
- Singkatan alamat dan koma sebelum tanggal diberi spasi: `KP.CIKANCUNG`, `GARUT,23-01-1988`.
- Nama kapital yang menempel dipecah bila setiap bagian ada di leksikon nama. String yang tidak dikenal dibiarkan utuh.
- Gelar di baris berikutnya, misalnya `S.Pd.I`, digabung ke `nama`.
- Label dari foto HP yang salah satu-dua huruf tetap dikenali, misalnya `Namá`, `Tempai/TgiLahir`, `Tempot/TgtLahr`, `Jens Kelamin`, `Ke/Desa`, `KeiDesa`, `Kacamatan`, `BerlakuHingga`, `Slatus Perkawinan`, dan `GolDarah`. Nilai di baris berikutnya tetap diambil, termasuk bila diawali titik dua.
- `LAKILAKI` menjadi `LAKI-LAKI`. Tanggal lahir yang memakai titik (`PEMALANG.08-09-1978`) diperlakukan seperti koma.
- `DESAPETANJUNGAN` dan `CIKARANGSELATAN` diberi spasi bila awalan atau akhiran tempatnya dikenal (`DESA`, `SELATAN`, dan sejenisnya).
- Tanggal terbit diambil dari tanggal paling bawah, bukan dari baris tempat/tanggal lahir.

Contoh dari KTP Garut (PP-OCRv6 small, `lang_type=id`):

```text
PROVINSIJAWA BARAT
KABUPATENGARUT
NIK
3205192301880001
Nama
:YOGIIRFANROSYADI,
S.Pd.I
Tempat/Tgl Lahir
:GARUT,23-01-1988
Alamat
:KP.CIKANCUNG
```

menjadi `nik=3205192301880001` dan `nama=YOGI IRFAN ROSYADI, S.Pd.I`.
