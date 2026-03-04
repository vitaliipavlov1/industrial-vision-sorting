# safety/latency_monitor.py

import time
import logging
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


class LatencyMonitor:
    """
    Real-time pipeline latency monitor.

    Uses a pre-allocated numpy circular buffer instead of deque
    to avoid per-call numpy array allocation from np.percentile(deque).

    At 120fps with window=200:
        deque approach: ~120 allocs/sec of 200-element arrays
        numpy buffer:   0 allocs/sec after __init__
    """

    def __init__(
        self,
        window_size:         int   = 200,
        max_latency_ms:      float = 30.0,
        overload_percentile: float = 95.0
    ):
        self._window_size         = window_size
        self._max_latency_ms      = max_latency_ms
        self._overload_percentile = overload_percentile

        # Pre-allocated circular buffer
        self._buf        = np.zeros(window_size, dtype=np.float64)
        self._head       = 0       # next write position
        self._count      = 0       # samples written (capped at window_size)

        self._last_start_ns: Optional[int] = None
        self._total_frames  = 0

    # ============================================================
    # FRAME TIMING
    # ============================================================

    def start_frame(self) -> None:
        self._last_start_ns = time.perf_counter_ns()

    def end_frame(self) -> None:
        if self._last_start_ns is None:
            return

        ms = (time.perf_counter_ns() - self._last_start_ns) / 1e6

        self._buf[self._head] = ms
        self._head  = (self._head + 1) % self._window_size
        self._count = min(self._count + 1, self._window_size)
        self._total_frames += 1
        self._last_start_ns = None

    def _window(self) -> np.ndarray:
        """Returns the valid window slice (no allocation for full buffer)."""
        if self._count < self._window_size:
            return self._buf[:self._count]
        return self._buf   # full buffer, no slice needed

    # ============================================================
    # METRICS
    # ============================================================

    def current_latency(self) -> float:
        if self._count == 0:
            return 0.0
        prev = (self._head - 1) % self._window_size
        return float(self._buf[prev])

    def average_latency(self) -> float:
        if self._count == 0:
            return 0.0
        return float(np.mean(self._window()))

    def p95_latency(self) -> float:
        if self._count == 0:
            return 0.0
        return float(np.percentile(self._window(), 95))

    def p99_latency(self) -> float:
        if self._count == 0:
            return 0.0
        return float(np.percentile(self._window(), 99))

    def max_latency(self) -> float:
        if self._count == 0:
            return 0.0
        return float(np.max(self._window()))

    def fps(self) -> float:
        avg = self.average_latency()
        return 1000.0 / avg if avg > 0 else 0.0

    def is_overloaded(self) -> bool:
        if self._count < 10:   # not enough samples yet
            return False
        return float(np.percentile(self._window(), self._overload_percentile)) \
               > self._max_latency_ms

    def stats(self) -> dict:
        return {
            "total_frames": self._total_frames,
            "window_size":  self._count,
            "current_ms":   self.current_latency(),
            "avg_ms":       self.average_latency(),
            "p95_ms":       self.p95_latency(),
            "p99_ms":       self.p99_latency(),
            "max_ms":       self.max_latency(),
            "fps":          self.fps(),
            "overloaded":   self.is_overloaded()
        }
