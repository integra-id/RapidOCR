# -*- encoding: utf-8 -*-
import threading
import time

import numpy as np
import pytest

from rapidocr.service.pool import EnginePool, build_pool
from rapidocr.service.preprocess import resize_max_side, scale_result_boxes


class _Engine:
    def __init__(self):
        self.calls = 0
        self.max_seen = 0
        self._lock = threading.Lock()

    def __call__(self, payload):
        with self._lock:
            self.calls += 1
            self.max_seen = max(self.max_seen, 1)
        return {"ok": True, "payload_is_array": isinstance(payload, np.ndarray)}


class _SlowEngine:
    def __init__(self, name):
        self.name = name
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def __call__(self, payload):
        with self._lock:
            self.calls += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            time.sleep(0.05)
            return self.name
        finally:
            with self._lock:
                self.concurrent -= 1


def test_build_pool_dedupes_the_same_engine_object():
    engine = _Engine()
    pool = build_pool(lambda: engine, size=4)
    assert pool.size == 1
    assert pool.unique == 1
    assert pool.run("x")["ok"] is True
    assert engine.calls == 1


def test_build_pool_keeps_distinct_sessions():
    engines = [_SlowEngine("a"), _SlowEngine("b"), _SlowEngine("c")]
    iterator = iter(engines)
    pool = build_pool(lambda: next(iterator), size=3)
    assert pool.size == 3
    assert pool.unique == 3
    results = pool.run_many(["one", "two", "three", "four"])
    assert sorted(results) == sorted(["a", "b", "c", "a"]) or len(results) == 4
    assert set(results) <= {"a", "b", "c"}
    assert sum(item.calls for item in engines) == 4
    assert max(item.max_concurrent for item in engines) == 1
    # At least two engines should have been used for four jobs.
    assert sum(1 for item in engines if item.calls) >= 2


def test_pool_run_many_is_parallel_with_distinct_engines():
    engines = [_SlowEngine("a"), _SlowEngine("b"), _SlowEngine("c"), _SlowEngine("d")]
    iterator = iter(engines)
    pool = build_pool(lambda: next(iterator), size=4)
    started = time.perf_counter()
    results = pool.run_many(["w", "x", "y", "z"])
    elapsed = time.perf_counter() - started
    assert sorted(results) == ["a", "b", "c", "d"]
    # Four 50ms jobs in parallel should be well under a sequential 200ms.
    assert elapsed < 0.18


def test_resize_max_side_and_box_scale():
    image = np.zeros((4096, 3072, 3), dtype=np.uint8)
    resized, scale_x, scale_y = resize_max_side(image, 1280)
    assert max(resized.shape[:2]) == 1280
    skinny = np.zeros((4096, 2304, 3), dtype=np.uint8)
    loosened, sx, sy = resize_max_side(skinny, 1280, min_short=960)
    assert max(loosened.shape[:2]) == 1707
    assert min(loosened.shape[:2]) == 960
    hard, _, _ = resize_max_side(skinny, 1280, min_short=0)
    assert max(hard.shape[:2]) == 1280
    assert scale_x == pytest.approx(3072 / resized.shape[1])
    assert scale_y == pytest.approx(4096 / resized.shape[0])

    class _Result:
        def __init__(self):
            self.boxes = np.array([[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]])

    result = scale_result_boxes(_Result(), scale_x, scale_y)
    assert result.boxes[0][1][0] == pytest.approx(10.0 * scale_x)

    same, sx, sy = resize_max_side(image[:100, :80], 1280)
    assert sx == 1.0 and sy == 1.0
    assert same.shape == (100, 80, 3)
