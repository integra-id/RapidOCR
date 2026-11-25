# -*- encoding: utf-8 -*-
"""Disk-backed OCR jobs.

Each job is a directory under the job root:

- ``job.json`` status
- ``input/`` uploaded bytes
- ``result.json`` when the job finishes

Directories older than ``RAPIDOCR_JOB_TTL_HOURS`` (default 24) are removed
when a job is created and when the worker starts.
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

from rapidocr.service.pdf import DocumentError

MAX_JOB_FILES = 20
MAX_ZIP_BYTES = 40 * 1024 * 1024
Runner = Callable[[list[tuple[str, bytes]], dict[str, Any]], dict[str, Any]]


def job_root() -> Path:
    configured = os.environ.get("RAPIDOCR_JOB_DIR")
    path = Path(configured) if configured else Path("/data/jobs")
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return path
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "rapidocr-jobs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def job_ttl_seconds() -> float:
    raw = os.environ.get("RAPIDOCR_JOB_TTL_HOURS", "24")
    try:
        hours = float(raw)
    except ValueError:
        hours = 24.0
    return max(0.0, hours) * 3600.0


class JobStore:
    def __init__(self, root: Path, runner: Runner):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self.cleanup()
        self._thread = threading.Thread(
            target=self._loop, name="rapidocr-jobs", daemon=True
        )
        self._thread.start()

    def create(
        self, uploads: list[tuple[str, bytes]], options: dict[str, Any]
    ) -> dict[str, Any]:
        self.cleanup()
        files = expand_uploads(uploads)
        job_id = uuid.uuid4().hex
        folder = self.root / job_id
        folder.mkdir(parents=True)
        incoming = folder / "input"
        incoming.mkdir()
        for index, (name, payload) in enumerate(files):
            safe = Path(name).name or f"file-{index}"
            (incoming / f"{index:02d}-{safe}").write_bytes(payload)
        record = {
            "id": job_id,
            "status": "queued",
            "task": options.get("task", "ocr"),
            "file_count": len(files),
            "created_at": time.time(),
            "error": None,
            "webhook_error": None,
        }
        self._write(folder / "job.json", record)
        (folder / "options.json").write_text(json.dumps(options), encoding="utf-8")
        self._queue.put(job_id)
        return record

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        path = self.root / job_id / "job.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def result(self, job_id: str) -> Optional[dict[str, Any]]:
        path = self.root / job_id / "result.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_inputs(self, job_id: str) -> list[tuple[str, bytes]]:
        incoming = self.root / job_id / "input"
        files = []
        for path in sorted(incoming.iterdir()):
            files.append((path.name.split("-", 1)[-1], path.read_bytes()))
        return files

    def cleanup(self) -> None:
        ttl = job_ttl_seconds()
        if ttl <= 0:
            return
        now = time.time()
        for folder in self.root.iterdir():
            meta = folder / "job.json"
            if not meta.is_file():
                continue
            try:
                created = json.loads(meta.read_text(encoding="utf-8")).get(
                    "created_at", 0
                )
            except json.JSONDecodeError:
                created = meta.stat().st_mtime
            if now - float(created) > ttl:
                for path in sorted(folder.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink(missing_ok=True)
                    elif path.is_dir():
                        path.rmdir()
                folder.rmdir()

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run(job_id)
            finally:
                self._queue.task_done()

    def _run(self, job_id: str) -> None:
        folder = self.root / job_id
        record = self.get(job_id)
        if record is None:
            return
        record["status"] = "running"
        self._write(folder / "job.json", record)
        try:
            options = json.loads((folder / "options.json").read_text(encoding="utf-8"))
            result = self.runner(self.load_inputs(job_id), options)
            self._write(folder / "result.json", result)
            record["status"] = "done"
            webhook = options.get("webhook")
            if webhook:
                record["webhook_error"] = _notify(webhook, job_id)
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
        self._write(folder / "job.json", record)

    def _write(self, path: Path, payload: dict[str, Any]) -> None:
        with self._lock:
            path.write_text(json.dumps(payload), encoding="utf-8")


def expand_uploads(uploads: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    for name, payload in uploads:
        if _is_zip(name, payload):
            files.extend(_unzip(payload))
        else:
            files.append((name or "upload", payload))
    if not files:
        raise DocumentError("The job has no files.")
    if len(files) > MAX_JOB_FILES:
        raise DocumentError(
            f"A job accepts at most {MAX_JOB_FILES} files.", status_code=413
        )
    return files


def _is_zip(name: str, payload: bytes) -> bool:
    return payload.startswith(b"PK") or name.lower().endswith(".zip")


def _unzip(payload: bytes) -> list[tuple[str, bytes]]:
    files = []
    total = 0
    try:
        archive = zipfile.ZipFile(BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise DocumentError("The upload is not a readable zip file.") from exc
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = Path(info.filename).name
        if not name or name.startswith("."):
            continue
        total += info.file_size
        if total > MAX_ZIP_BYTES:
            raise DocumentError(
                "Uncompressed zip is larger than 40 MB.", status_code=413
            )
        files.append((name, archive.read(info)))
    return files


def _notify(url: str, job_id: str) -> Optional[str]:
    try:
        import requests

        response = requests.post(url, json={"id": job_id, "status": "done"}, timeout=5)
        response.raise_for_status()
    except Exception as exc:
        return str(exc)
    return None
