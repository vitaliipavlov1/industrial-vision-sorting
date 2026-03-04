# prediction/predictor.py

import logging
from typing import Optional

from core.models import TrackState, FireEvent
from core.interfaces import PredictorInterface


logger = logging.getLogger(__name__)


class Predictor(PredictorInterface):
    """
    Conveyor ejector fire time predictor.

    Computes the precise moment to open the solenoid valve
    so that the air jet intersects the object exactly at the nozzle.

    Timing model:
        time_to_nozzle = distance_to_nozzle / object_velocity
        fire_time      = now + time_to_nozzle - (valve_delay + air_response + safety_margin)

    Velocity source priority:
        1. Kalman-filtered track velocity  (from camera + tracker)
        2. Encoder velocity fallback        (when track velocity unreliable)

    Key improvement over previous version:
        - Encoder velocity fallback when track is young / low-confidence
        - Per-class fire duration override support
        - Fire event deduplication: same object only scheduled once per
          approach (guarded by confirmed_object_ids in caller)
        - Detailed rejection logging for tuning
    """

    def __init__(
        self,
        nozzle_position_mm:       float,
        valve_delay_ms:           float,
        air_response_ms:          float,
        fire_duration_ms:         int,
        min_velocity_mm_s:        float,
        safety_margin_ms:         float,
        min_distance_mm:          float,
        encoder_velocity_fallback: bool  = True,
        per_class_duration_ms:    Optional[dict] = None
    ):

        self._nozzle_position_mm       = nozzle_position_mm
        self._valve_delay_ms           = valve_delay_ms
        self._air_response_ms          = air_response_ms
        self._fire_duration_ms         = fire_duration_ms
        self._min_velocity_mm_s        = min_velocity_mm_s
        self._safety_margin_ms         = safety_margin_ms
        self._min_distance_mm          = min_distance_mm
        self._encoder_velocity_fallback = encoder_velocity_fallback
        self._per_class_duration_ms    = per_class_duration_ms or {}

        self._total_delay_ms = (
            valve_delay_ms + air_response_ms + safety_margin_ms
        )

        logger.info(
            "Predictor initialized: nozzle=%.1fmm  "
            "total_delay=%.1fms  fire_duration=%dms",
            nozzle_position_mm, self._total_delay_ms, fire_duration_ms
        )

    # ============================================================
    # COMPUTE FIRE EVENT
    # ============================================================

    def compute_fire_event(
        self,
        track:               TrackState,
        current_time_ns:     int,
        encoder_velocity_mm_s: float = 0.0
    ) -> Optional[FireEvent]:
        """
        Computes fire time for a confirmed track.

        Args:
            track:                 Current track state
            current_time_ns:       Current time (perf_counter_ns)
            encoder_velocity_mm_s: Belt encoder velocity as fallback

        Returns:
            FireEvent or None if firing is not appropriate.
        """

        # --------------------------------------------------------
        # Velocity selection
        # --------------------------------------------------------

        velocity = track.velocity_mm_s

        if abs(velocity) < self._min_velocity_mm_s:

            # Fall back to encoder if track velocity is unreliable
            if self._encoder_velocity_fallback and abs(encoder_velocity_mm_s) >= self._min_velocity_mm_s:

                velocity = encoder_velocity_mm_s

                logger.debug(
                    "Predictor: object #%d using encoder velocity "
                    "fallback (%.1f mm/s)",
                    track.object_id, velocity
                )

            else:

                logger.debug(
                    "Predictor: object #%d rejected — velocity "
                    "%.1f mm/s below min %.1f mm/s",
                    track.object_id,
                    track.velocity_mm_s,
                    self._min_velocity_mm_s
                )

                return None

        # --------------------------------------------------------
        # Distance check
        # --------------------------------------------------------

        distance_to_nozzle = self._nozzle_position_mm - track.x_mm

        if distance_to_nozzle < self._min_distance_mm:

            logger.debug(
                "Predictor: object #%d rejected — "
                "distance %.1fmm < min %.1fmm",
                track.object_id, distance_to_nozzle, self._min_distance_mm
            )

            return None

        # --------------------------------------------------------
        # Timing calculation
        # --------------------------------------------------------

        time_to_nozzle_ms  = (distance_to_nozzle / velocity) * 1000.0
        fire_time_offset_ms = time_to_nozzle_ms - self._total_delay_ms

        if fire_time_offset_ms <= 0:

            logger.debug(
                "Predictor: object #%d rejected — "
                "fire offset %.1fms <= 0 (too close or too fast)",
                track.object_id, fire_time_offset_ms
            )

            return None

        fire_time_ns = current_time_ns + int(fire_time_offset_ms * 1e6)

        # --------------------------------------------------------
        # Per-class duration override
        # --------------------------------------------------------

        duration_ms = self._per_class_duration_ms.get(
            track.class_id,
            self._fire_duration_ms
        )

        logger.debug(
            "Predictor: object #%d  class=%d  "
            "dist=%.1fmm  vel=%.1fmm/s  "
            "t_to_nozzle=%.1fms  fire_offset=%.1fms  dur=%dms",
            track.object_id, track.class_id,
            distance_to_nozzle, velocity,
            time_to_nozzle_ms, fire_time_offset_ms, duration_ms
        )

        return FireEvent(
            object_id    = track.object_id,
            class_id     = track.class_id,
            fire_time_ns = fire_time_ns,
            duration_ms  = duration_ms
        )
