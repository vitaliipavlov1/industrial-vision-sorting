# safety/watchdog.py

import time
import logging
import threading
from typing import Optional, Callable

from core.models import SystemFault, SystemFaultCode
from core.interfaces import FaultHandlerInterface


logger = logging.getLogger(__name__)


# ============================================================
# FAULT HANDLER
# ============================================================

class LoggingFaultHandler(FaultHandlerInterface):

    def __init__(self, max_history: int = 1000):
        self._history = []
        self._max     = max_history
        self._lock    = threading.Lock()

    def report(self, fault: SystemFault) -> None:
        with self._lock:
            self._history.append(fault)
            if len(self._history) > self._max:
                self._history.pop(0)
        logger.error("FAULT [%s] %s", fault.code.name, fault.message)

    def get_history(self):
        with self._lock:
            return list(self._history)


# ============================================================
# WATCHDOG
# ============================================================

class Watchdog:
    """
    Industrial safety watchdog.

    Thread safety fix:
        _last_frame_time is written by main pipeline thread and
        read by the watchdog thread. Uses threading.Event + atomic
        int write pattern (assignment of int is atomic in CPython,
        but we use a Lock to be explicit and portable).
    """

    def __init__(
        self,
        latency_monitor,
        ejector,
        tracker,
        fault_handler:       Optional[FaultHandlerInterface] = None,
        frame_timeout_ms:    float = 200.0,
        check_interval_ms:   float = 50.0,
        recovery_timeout_s:  float = 5.0
    ):
        self._latency_monitor   = latency_monitor
        self._ejector           = ejector
        self._tracker           = tracker
        self._fault_handler     = fault_handler or LoggingFaultHandler()

        self._frame_timeout_ns  = int(frame_timeout_ms * 1e6)
        self._check_interval    = check_interval_ms / 1000.0
        self._recovery_timeout_s = recovery_timeout_s

        # Thread-safe frame timestamp
        self._frame_time_lock   = threading.Lock()
        self._last_frame_time   = time.perf_counter_ns()

        self._fault_active      = False
        self._fault_code        = SystemFaultCode.NONE
        self._fault_since_ns    = 0

        self._running  = False
        self._thread:  Optional[threading.Thread] = None

        self.on_fault:    Optional[Callable] = None
        self.on_recovery: Optional[Callable] = None

    # ============================================================
    # HEARTBEAT
    # ============================================================

    def notify_frame(self) -> None:
        """Called by main thread on every processed frame."""
        with self._frame_time_lock:
            self._last_frame_time = time.perf_counter_ns()

    def _get_last_frame_time(self) -> int:
        with self._frame_time_lock:
            return self._last_frame_time

    # ============================================================
    # START / STOP
    # ============================================================

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="watchdog"
        )
        self._thread.start()
        logger.info("Watchdog started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Watchdog stopped")

    # ============================================================
    # MAIN LOOP
    # ============================================================

    def _loop(self) -> None:

        while self._running:

            now              = time.perf_counter_ns()
            last_frame_time  = self._get_last_frame_time()
            frame_elapsed_ns = now - last_frame_time

            if frame_elapsed_ns > self._frame_timeout_ns:
                self._trigger_fault(
                    SystemFaultCode.FRAME_TIMEOUT,
                    f"Frame timeout: {frame_elapsed_ns/1e6:.1f}ms "
                    f"(threshold {self._frame_timeout_ns/1e6:.0f}ms)"
                )

            elif self._latency_monitor.is_overloaded():
                self._trigger_fault(
                    SystemFaultCode.LATENCY_OVERLOAD,
                    f"Latency overload: p95={self._latency_monitor.p95_latency():.1f}ms  "
                    f"fps={self._latency_monitor.fps():.1f}"
                )

            elif self._fault_active:
                elapsed_s = (now - self._fault_since_ns) / 1e9
                if elapsed_s >= self._recovery_timeout_s:
                    self._attempt_recovery()

            time.sleep(self._check_interval)

    # ============================================================
    # FAULT
    # ============================================================

    def _trigger_fault(self, code: SystemFaultCode, message: str) -> None:

        if self._fault_active and self._fault_code == code:
            return   # de-duplicate

        self._fault_active   = True
        self._fault_code     = code
        self._fault_since_ns = time.perf_counter_ns()

        self._ejector.disable()
        self._tracker.reset()

        fault = SystemFault(
            code         = code,
            message      = message,
            timestamp_ns = self._fault_since_ns,
            context      = {
                "avg_latency_ms": self._latency_monitor.average_latency(),
                "fps":            self._latency_monitor.fps()
            }
        )

        self._fault_handler.report(fault)

        if self.on_fault:
            try:
                self.on_fault(fault)
            except Exception as e:
                logger.error("on_fault callback error: %s", e)

    # ============================================================
    # RECOVERY
    # ============================================================

    def _attempt_recovery(self) -> None:

        if self._latency_monitor.is_overloaded():
            return

        if time.perf_counter_ns() - self._get_last_frame_time() > self._frame_timeout_ns:
            return

        self._ejector.enable()
        self._fault_active = False
        self._fault_code   = SystemFaultCode.NONE

        elapsed = (time.perf_counter_ns() - self._fault_since_ns) / 1e9
        logger.info("Watchdog: recovered after %.1fs — ejector re-enabled", elapsed)

        if self.on_recovery:
            try:
                self.on_recovery()
            except Exception as e:
                logger.error("on_recovery callback error: %s", e)

    # ============================================================
    # STATUS
    # ============================================================

    @property
    def fault_active(self) -> bool:
        return self._fault_active

    @property
    def fault_code(self) -> SystemFaultCode:
        return self._fault_code

    def get_fault_history(self):
        return self._fault_handler.get_history()
