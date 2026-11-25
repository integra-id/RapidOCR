# -*- encoding: utf-8 -*-
"""Optional API key and a small in-memory rate limit.

Auth is off when ``RAPIDOCR_API_KEY`` is empty. ``/health`` stays open.
``RAPIDOCR_RATE_LIMIT`` is requests per minute for each key, or each client
when no key is configured. ``0`` disables the limit. The default is 60.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI
from fastapi.responses import JSONResponse

_HITS: dict[str, deque] = defaultdict(deque)
_OPEN_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


def reset_rate_limits() -> None:
    _HITS.clear()


def install_guard(app: FastAPI) -> None:
    @app.middleware("http")
    async def guard(request, call_next):
        path = request.url.path
        if path in _OPEN_PATHS:
            return await call_next(request)

        configured = os.environ.get("RAPIDOCR_API_KEY", "").strip()
        provided = _presented_key(request)
        if configured and not _same(provided, configured):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key."},
                media_type="application/json",
            )

        limit = _rate_limit()
        if limit > 0:
            identity = provided or _client_host(request)
            if not _allow(identity, limit):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded."},
                    media_type="application/json",
                )
        return await call_next(request)


def _presented_key(request) -> str:
    header = request.headers.get("x-api-key", "").strip()
    if header:
        return header
    authorization = request.headers.get("authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def _same(provided: str, expected: str) -> bool:
    if len(provided) != len(expected):
        return False
    mismatch = 0
    for left, right in zip(provided, expected):
        mismatch |= ord(left) ^ ord(right)
    return mismatch == 0


def _client_host(request) -> str:
    if request.client is None:
        return "anonymous"
    return request.client.host or "anonymous"


def _rate_limit() -> int:
    raw = os.environ.get("RAPIDOCR_RATE_LIMIT", "60").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 60


def _allow(identity: str, limit: int) -> bool:
    now = time.monotonic()
    hits = _HITS[identity]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= limit:
        return False
    hits.append(now)
    return True
