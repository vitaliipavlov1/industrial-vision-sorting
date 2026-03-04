# core/double_buffer.py

import threading


class DoubleBuffer:
    """
    Thread-safe double buffer for real-time GPU/CPU pipeline overlap.

    Allows overlapping GPU and CPU stages:
        Frame N   → GPU  (writing to buffer A)
        Frame N-1 → CPU  (reading from buffer B)

    Thread safety:
        next() and current() are protected by a Lock.
        In practice, DoubleBuffer is only accessed from the single
        pipeline thread in this project, but the Lock makes the
        contract explicit and safe for future multi-threaded use.
    """

    def __init__(self, buffer_a, buffer_b):

        self._buffers = [buffer_a, buffer_b]
        self._index   = 0
        self._lock    = threading.Lock()

    def current(self):
        """Returns the currently active buffer without switching."""
        with self._lock:
            return self._buffers[self._index]

    def next(self):
        """Switches to the other buffer and returns it."""
        with self._lock:
            self._index ^= 1
            return self._buffers[self._index]

    def all(self):
        """Returns both buffers (for cleanup)."""
        return self._buffers
