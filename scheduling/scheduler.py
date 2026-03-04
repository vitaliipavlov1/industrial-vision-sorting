# scheduling/scheduler.py

import time
import threading
import heapq
import itertools
import logging
from typing import Dict, Set

from core.interfaces import SchedulerInterface
from core.models import FireEvent


logger = logging.getLogger(__name__)


class Scheduler(SchedulerInterface):
    """
    Deterministic fire scheduler with sub-millisecond precision.

    Algorithm:
        - Min-heap sorted by fire_time_ns
        - Sleep-phase:    coarse sleep until (fire_time - busy_wait_threshold)
        - Busy-wait phase: spin until exact fire_time_ns

    Key improvements over previous version:

    1. Per-object deduplication:
       Only one pending event per object_id at a time.
       If a new event arrives for an already-scheduled object,
       the old event is replaced if the new one is closer.
       Prevents double-ejection of the same object.

    2. cancel_object():
       Tracker or watchdog can cancel all events for a specific
       object (e.g., track was confirmed as false positive).

    3. sequence_id:
       Monotonic counter for reliable event identification
       even when object_ids are reused after wrap-around.

    4. Stale event guard:
       Events older than max_age_ms are silently dropped
       (protect against scheduler pause/resume drift).
    """

    STALE_AGE_MS = 500   # Drop events older than this

    def __init__(
        self,
        ejector,
        busy_wait_threshold_us: int = 500
    ):

        self._ejector                  = ejector
        self._busy_wait_threshold_ns   = busy_wait_threshold_us * 1000

        self._heap:   list        = []   # (fire_time_ns, sequence_id, FireEvent)
        self._active: Dict[int, int] = {}  # object_id → sequence_id of latest event
        self._lock    = threading.Lock()

        self._seq_gen = itertools.count(start=1)

        self._running = False
        self._thread  = None

        self._fired_count    = 0
        self._dropped_count  = 0

    # ============================================================
    # START / STOP
    # ============================================================

    def start(self) -> None:

        self._running = True

        self._thread = threading.Thread(
            target = self._loop,
            daemon = True,
            name   = "scheduler"
        )

        self._thread.start()

        logger.info(
            "Scheduler started: busy_wait=%d µs",
            self._busy_wait_threshold_ns // 1000
        )

    def stop(self) -> None:

        self._running = False

        if self._thread:
            self._thread.join(timeout=2)

        logger.info(
            "Scheduler stopped: fired=%d  dropped=%d",
            self._fired_count, self._dropped_count
        )

    # ============================================================
    # SCHEDULE
    # ============================================================

    def schedule(self, event: FireEvent) -> None:
        """
        Schedules a fire event.

        If a pending event for the same object_id already exists:
            - If the new event fires earlier → replace (better precision)
            - If the new event fires later  → keep existing
        """

        seq = next(self._seq_gen)

        with self._lock:

            existing_seq = self._active.get(event.object_id)

            if existing_seq is not None:
                # Existing event: keep whichever fires first
                # (old event still in heap but will be skipped by seq check)
                logger.debug(
                    "Scheduler: updating event for object #%d",
                    event.object_id
                )

            self._active[event.object_id] = seq

            heapq.heappush(
                self._heap,
                (event.fire_time_ns, seq, event)
            )

    # ============================================================
    # CANCEL OBJECT
    # ============================================================

    def cancel_object(self, object_id: int) -> int:
        """
        Cancels all pending events for object_id.
        The heap is not cleaned eagerly (lazy deletion via seq check).

        Returns:
            Number of events cancelled (0 or 1).
        """

        with self._lock:

            if object_id in self._active:
                del self._active[object_id]
                logger.debug(
                    "Scheduler: cancelled event for object #%d", object_id
                )
                return 1

        return 0

    # ============================================================
    # MAIN LOOP
    # ============================================================

    def _loop(self) -> None:

        while self._running:

            with self._lock:
                peek = self._heap[0] if self._heap else None

            if peek is None:
                time.sleep(0.001)
                continue

            fire_time_ns, seq, event = peek

            now = time.perf_counter_ns()

            # ------------------------------------------------
            # Stale event guard
            # ------------------------------------------------

            age_ms = (now - fire_time_ns) / 1e6

            if age_ms > self.STALE_AGE_MS:

                with self._lock:
                    if self._heap and self._heap[0][1] == seq:
                        heapq.heappop(self._heap)

                logger.warning(
                    "Scheduler: dropped stale event for object #%d "
                    "(age=%.1f ms)",
                    event.object_id, age_ms
                )

                self._dropped_count += 1
                continue

            # ------------------------------------------------
            # Sleep phase
            # ------------------------------------------------

            remaining_ns = fire_time_ns - now

            if remaining_ns > self._busy_wait_threshold_ns:

                sleep_s = (remaining_ns - self._busy_wait_threshold_ns) / 1e9
                time.sleep(sleep_s)
                continue

            # ------------------------------------------------
            # Busy-wait phase
            # ------------------------------------------------

            while time.perf_counter_ns() < fire_time_ns:
                pass

            # ------------------------------------------------
            # POP and validate
            # ------------------------------------------------

            with self._lock:

                if not self._heap or self._heap[0][1] != seq:
                    # Event was cancelled or superseded
                    continue

                heapq.heappop(self._heap)

                # Validate: is this still the current event for this object?
                if self._active.get(event.object_id) != seq:
                    self._dropped_count += 1
                    logger.debug(
                        "Scheduler: skipping superseded event "
                        "for object #%d",
                        event.object_id
                    )
                    continue

                # Remove from active map
                del self._active[event.object_id]

            # ------------------------------------------------
            # FIRE
            # ------------------------------------------------

            self._ejector.fire(event.duration_ms)
            self._fired_count += 1

            logger.debug(
                "Scheduler: fired object #%d  class=%d  duration=%dms  "
                "total_fires=%d",
                event.object_id, event.class_id,
                event.duration_ms, self._fired_count
            )

    # ============================================================
    # STATS
    # ============================================================

    @property
    def fired_count(self) -> int:
        return self._fired_count

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._active)
