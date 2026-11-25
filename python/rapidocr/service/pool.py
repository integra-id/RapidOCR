# -*- encoding: utf-8 -*-
"""A pool of RapidOCR engine sessions for concurrent in-process work."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Optional, Sequence

from rapidocr.service.runtime import worker_count


class EnginePool:
    """Hand out one engine per request; several distinct engines can run at once."""

    def __init__(self, engines: Sequence[Any]):
        if not engines:
            raise ValueError("EnginePool needs at least one engine.")
        self._engines = list(engines)
        self.size = len(self._engines)
        self.unique = len({id(item) for item in self._engines})
        self._queue: queue.Queue = queue.Queue()
        for engine in self._engines:
            self._queue.put(engine)
        # The same object must not run in two threads at once (tests, or a
        # loader that returns a cached engine).
        self._lock = threading.Lock() if self.unique == 1 else None

    def __call__(self, payload):
        return self.run(payload)

    def run(self, payload):
        engine = self._queue.get()
        try:
            if self._lock is not None:
                with self._lock:
                    return engine(payload)
            return engine(payload)
        finally:
            self._queue.put(engine)

    def run_many(self, images: Iterable[Any]) -> list:
        items = list(images)
        if not items:
            return []
        workers = min(len(items), self.unique)
        if workers <= 1:
            return [self.run(item) for item in items]
        results: list = [None] * len(items)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.run, item): index for index, item in enumerate(items)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results


def build_pool(loader: Callable[[], Any], size: Optional[int] = None) -> EnginePool:
    """Create ``size`` engines, or one if ``loader`` returns the same object."""
    count = size if size is not None else worker_count()
    count = max(1, int(count))
    first = loader()
    engines = [first]
    for _ in range(count - 1):
        extra = loader()
        if extra is first:
            break
        engines.append(extra)
    return EnginePool(engines)
