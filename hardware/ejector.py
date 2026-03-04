# hardware/ejector.py

import time
import queue
import threading
import logging

from core.interfaces import EjectorInterface


logger = logging.getLogger(__name__)


class Ejector(EjectorInterface):
    """
    Industrial ejector controller.

    Controls solenoid valve / air nozzle / reject actuator.

    Design:
        fire() is non-blocking — enqueues pulse into dedicated
        worker thread and returns immediately. This ensures the
        Scheduler thread is never blocked for pulse duration.

        Worker thread executes pulses sequentially to prevent
        solenoid overlap (hardware protection).
    """

    MAX_DURATION_MS = 200

    def __init__(
        self,
        gpio_interface,
        output_pin:      int,
        max_duration_ms: int = 200
    ):
        self._gpio           = gpio_interface
        self._pin            = output_pin
        self._max_duration_ms = min(max_duration_ms, self.MAX_DURATION_MS)

        self._enabled    = True
        self._lock       = threading.Lock()
        self._fire_count = 0
        self._active     = False

        self._fire_queue = queue.Queue(maxsize=16)

        self._fire_thread = threading.Thread(
            target=self._fire_worker,
            daemon=True,
            name="ejector-fire"
        )
        self._fire_thread.start()

        self._gpio.setup(self._pin, "output")
        self._gpio.write(self._pin, 0)

        logger.info(
            "Ejector initialized: pin=%d  max_duration=%dms",
            output_pin, self._max_duration_ms
        )

    # ============================================================
    # FIRE  (non-blocking)
    # ============================================================

    def fire(self, duration_ms: int) -> None:

        if not self._enabled:
            logger.debug("Ejector.fire() ignored: disabled")
            return

        clamped = min(duration_ms, self._max_duration_ms)

        if clamped != duration_ms:
            logger.warning(
                "Ejector: duration %dms clamped to %dms",
                duration_ms, clamped
            )

        try:
            self._fire_queue.put_nowait(clamped)
        except queue.Full:
            logger.warning(
                "Ejector: fire queue full — pulse dropped (scheduler overload)"
            )

    # ============================================================
    # FIRE WORKER THREAD
    # ============================================================

    def _fire_worker(self) -> None:

        while True:

            try:
                duration_ms = self._fire_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._enabled:
                continue

            try:
                with self._lock:
                    if not self._enabled:
                        continue
                    self._active = True
                    self._gpio.write(self._pin, 1)

                time.sleep(duration_ms / 1000.0)

                with self._lock:
                    self._gpio.write(self._pin, 0)
                    self._active     = False
                    self._fire_count += 1

                logger.debug(
                    "Ejector fired: %dms  total=%d",
                    duration_ms, self._fire_count
                )

            except Exception as exc:
                logger.error("Ejector fire error: %s", exc)
                try:
                    self._gpio.write(self._pin, 0)
                    self._active = False
                except Exception:
                    pass

    # ============================================================
    # DISABLE / ENABLE
    # ============================================================

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
            self._gpio.write(self._pin, 0)
            self._active  = False

        drained = 0
        while not self._fire_queue.empty():
            try:
                self._fire_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break

        logger.warning("Ejector DISABLED. Drained %d pending events.", drained)

    def enable(self) -> None:
        with self._lock:
            self._enabled = True
        logger.info("Ejector ENABLED")

    # ============================================================
    # PROPERTIES
    # ============================================================

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def fire_count(self) -> int:
        return self._fire_count

    @property
    def is_active(self) -> bool:
        return self._active
