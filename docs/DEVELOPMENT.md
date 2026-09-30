# Development branch

Ongoing work for this fork lives on `development`. `main` stays the integration branch and the place to pull upstream RapidOCR updates. Open pull requests from `development` into `main`.

## Default OCR language

Bahasa Indonesia (`id`) is the default `Det.lang_type` and `Rec.lang_type`. The text-line angle classifier stays `ch` because that checkpoint is shared and has no per-language variant.

The default PP-OCRv6 pipeline keeps one multilingual detector, recognizer, and character dictionary. Indonesian is the selected language, not the only language. Pass `LangDet` / `LangRec` or a `lang_type` string (`ch`, `en`, `japan`, and the rest) to use another language.

PP-OCRv4 and PP-OCRv5 have no separate Indonesian checkpoint. `id` (and aliases such as `indonesian` and `bahasa_indonesia`) is wired to the Latin recognition model and dictionary. Detection uses the multilingual detector on v4 and the Chinese detector on v5.

KTP fields are parsed after OCR by `parse_ktp` (`python/rapidocr/postprocess/ktp.py`). Usage is in [ktp.md](./ktp.md). The parser does not replace other-language OCR paths.

## Merge `main` or upstream into `development`

Keep history mergeable. Do not rebase or otherwise rewrite commits that are already on `development`.

From this fork's `main`:

```bash
git checkout development
git fetch origin
git merge origin/main
```

From upstream [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR):

```bash
git remote add upstream https://github.com/RapidAI/RapidOCR.git  # once
git fetch upstream
git checkout development
git merge upstream/main
```

Resolve conflicts, keep the Indonesian defaults unless the same lines were intentionally changed upstream, then push the merge commit:

```bash
git push origin development
```

## Menggabungkan `main` ke `development`

Kerja lanjutan ada di branch `development`. Jangan rebase commit yang sudah ter-push. Gunakan merge biasa supaya `main` tetap bisa digabung lagi nanti.

```bash
git checkout development
git fetch origin
git merge origin/main
git push origin development
```

Untuk mengambil pembaruan dari upstream RapidAI/RapidOCR, tambahkan remote `upstream` sekali, lalu `git fetch upstream` dan `git merge upstream/main` saat berada di `development`.
