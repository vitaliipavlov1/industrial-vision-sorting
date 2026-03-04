# capture/camera.py

import threading
import time
import logging
from queue import Queue, Full

import numpy as np

from core.models import Frame


logger = logging.getLogger(__name__)


class CameraThread:
    """
    Real-time camera capture thread.

    Features:
        - Frame drop strategy: drops oldest frame when queue is full
          (always delivers latest frame to pipeline)
        - Monotonic frame_id counter for tracking frame loss
        - Encoder timestamp synchronization
        - Contiguous memory guarantee (np.ascontiguousarray)
          required by CUDA async memcpy
        - Graceful error handling with cooldown on repeated failures
    """

    def __init__(
        self,
        camera,
        encoder,
        output_queue:   Queue,
        drop_log_every: int = 100
    ):

        self._camera         = camera
        self._encoder        = encoder
        self._output_queue   = output_queue
        self._drop_log_every = drop_log_every

        self._running  = False
        self._thread   = None

        self._frame_id     = 0
        self._dropped      = 0
        self._errors       = 0

    # ============================================================
    # START
    # ============================================================

    def start_capture(self) -> None:

        self._running = True
        self._camera.start()

        self._thread = threading.Thread(
            target = self._capture_loop,
            daemon = True,
            name   = "camera-capture"
        )

        self._thread.start()

        logger.info("CameraThread started")

    # ============================================================
    # STOP
    # ============================================================

    def stop_capture(self) -> None:

        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2)

        self._camera.stop()

        logger.info(
            "CameraThread stopped: frames=%d  dropped=%d  errors=%d",
            self._frame_id, self._dropped, self._errors
        )

    # ============================================================
    # CAPTURE LOOP
    # ============================================================

    def _capture_loop(self) -> None:

        while self._running:

            # ------------------------------------------------
            # Grab frame from camera driver
            # ------------------------------------------------

            try:
                frame_img = self._camera.grab_frame()
            except Exception as exc:
                self._errors += 1
                logger.error(
                    "CameraThread: grab_frame error #%d: %s",
                    self._errors, exc
                )
                time.sleep(0.005)
                continue

            if frame_img is None:
                continue

            # ------------------------------------------------
            # Ensure C-contiguous layout for CUDA async memcpy
            # ------------------------------------------------

            if not frame_img.flags["C_CONTIGUOUS"]:
                frame_img = np.ascontiguousarray(frame_img)

            # ------------------------------------------------
            # Timestamp + encoder position
            # ------------------------------------------------

            timestamp_ns      = time.perf_counter_ns()
            belt_position_mm  = 0.0

            if self._encoder is not None:
                try:
                    belt_position_mm = self._encoder.get_position_mm()
                except Exception as exc:
                    logger.warning(
                        "CameraThread: encoder read failed: %s", exc
                    )

            self._frame_id += 1

            frame = Frame(
                image            = frame_img,
                timestamp_ns     = timestamp_ns,
                belt_position_mm = belt_position_mm,
                frame_id         = self._frame_id
            )

            # ------------------------------------------------
            # Frame drop strategy: evict oldest, enqueue newest
            # ------------------------------------------------

            if self._output_queue.full():

                try:
                    self._output_queue.get_nowait()
                    self._dropped += 1

                    if self._dropped % self._drop_log_every == 0:
                        logger.warning(
                            "CameraThread: %d frames dropped (queue full). "
                            "Pipeline may be too slow.",
                            self._dropped
                        )
                except Exception:
                    pass

            try:
                self._output_queue.put_nowait(frame)
            except Full:
                self._dropped += 1

    # ============================================================
    # STATUS
    # ============================================================

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_id

    @property
    def drop_count(self) -> int:
        return self._dropped
