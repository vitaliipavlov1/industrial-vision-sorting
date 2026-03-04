# core/pinned_memory.py

import logging

import numpy as np
import cupy as cp
import cupy.cuda


logger = logging.getLogger(__name__)


class PinnedMemory:
    """
    Pinned (page-locked) host memory buffer backed by CuPy.

    Advantages over pageable memory:
        - faster GPU ↔ CPU transfers (DMA without staging)
        - avoids page faults during cudaMemcpyAsync
        - stable, predictable latency for real-time pipelines
        - allocated once and reused across all frames

    CuPy replaces manual cudaHostAlloc + ctypes pointer arithmetic.
    """

    def __init__(self, size: int, dtype=np.uint8):

        self.size      = size
        self.dtype     = dtype
        self.itemsize  = np.dtype(dtype).itemsize
        self.nbytes    = size * self.itemsize

        # Allocate pinned memory via CuPy
        self._pinned_mem = cp.cuda.alloc_pinned_memory(self.nbytes)

        # Zero-copy numpy view over the pinned buffer
        self.array = np.frombuffer(self._pinned_mem, dtype=dtype, count=size)

        logger.debug(
            "PinnedMemory allocated: %d elements × %d bytes = %d KB",
            size, self.itemsize, self.nbytes // 1024
        )

    # ============================================================
    # ACCESS
    # ============================================================

    def get_numpy(self, shape: tuple) -> np.ndarray:
        """Returns a numpy view of the pinned buffer reshaped to shape."""
        return self.array.reshape(shape)

    # ============================================================
    # POINTER  (for TensorRT / low-level interop)
    # ============================================================

    def device_pointer(self) -> int:
        """Returns the raw host pointer as int (for CUDA interop)."""
        return self._pinned_mem.ptr

    # ============================================================
    # RESET
    # ============================================================

    def zero(self) -> None:
        """Clears buffer without reallocation."""
        self.array[:] = 0

    # ============================================================
    # FREE
    # ============================================================

    def free(self) -> None:
        """Explicitly release pinned memory."""
        if self._pinned_mem is not None:
            # CuPy PinnedMemory is reference-counted and freed automatically,
            # but we release our reference explicitly for deterministic cleanup.
            self._pinned_mem = None
            logger.debug("PinnedMemory freed")

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass
