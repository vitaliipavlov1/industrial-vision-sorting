# tracking/tracker.py

import itertools
import logging
from typing import List, Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment  # module-level import

from core.models import Detection, TrackState
from core.interfaces import TrackerInterface


logger = logging.getLogger(__name__)


# ============================================================
# INTERNAL TRACK  (1D Kalman filter)
# ============================================================

class _Track:
    """
    1D Kalman filter track for conveyor objects.

    State:       [x_mm, velocity_mm_s]
    Measurement: [x_mm]
    """

    __slots__ = (
        "track_id", "class_id", "area_mm2", "y_mm",
        "timestamp_ns", "belt_position_mm",
        "missed_frames", "age_frames", "confirmed",
        "x_kf", "P", "Q", "R", "H"
    )

    def __init__(
        self,
        track_id:          int,
        detection:         Detection,
        process_noise:     float,
        measurement_noise: float
    ):
        self.track_id         = track_id
        self.class_id         = detection.class_id
        self.area_mm2         = detection.area_mm2
        self.y_mm             = detection.y_mm
        self.timestamp_ns     = detection.timestamp_ns
        self.belt_position_mm = 0.0
        self.missed_frames    = 0
        self.age_frames       = 1
        self.confirmed        = False

        self.x_kf = np.array([detection.x_mm, 0.0], dtype=np.float64)

        self.P = np.diag([
            measurement_noise ** 2,
            500.0 ** 2
        ])
        self.Q = np.diag([
            (process_noise * 0.5) ** 2,
            (process_noise * 50.0) ** 2
        ])
        self.R = np.array([[measurement_noise ** 2]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)

    def predict(self, dt: float) -> None:
        if dt <= 0.0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        self.x_kf = F @ self.x_kf
        self.P    = F @ self.P @ F.T + self.Q

    def update(self, detection: Detection, dt: float) -> None:
        self.predict(dt)

        z = np.array([[detection.x_mm]], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x_kf = self.x_kf + (K @ (z - self.H @ self.x_kf)).ravel()
        self.P    = (np.eye(2) - K @ self.H) @ self.P

        self.y_mm          = detection.y_mm
        self.area_mm2      = 0.7 * self.area_mm2 + 0.3 * detection.area_mm2
        self.timestamp_ns  = detection.timestamp_ns
        self.missed_frames = 0
        self.age_frames   += 1

    @property
    def x_mm(self) -> float:
        return self.x_kf[0]

    @property
    def velocity_mm_s(self) -> float:
        return self.x_kf[1]


# ============================================================
# TRACKER
# ============================================================

class Tracker(TrackerInterface):
    """
    Industrial conveyor multi-object tracker.

    Features:
        - 1D Kalman filter per track
        - Hungarian assignment (scipy, module-level import)
        - O(1) track lookup via dict
        - min_confirmations before firing allowed
        - encoder velocity as prior for new tracks
    """

    def __init__(
        self,
        process_noise:            float,
        measurement_noise:        float,
        base_gating_mm:           float,
        max_missed_frames:        int,
        velocity_smoothing_alpha: float,   # unused — Kalman handles it
        min_confirmations:        int = 3
    ):
        self._process_noise     = process_noise
        self._measurement_noise = measurement_noise
        self._base_gating_mm    = base_gating_mm
        self._max_missed_frames = max_missed_frames
        self._min_confirmations = min_confirmations

        # O(1) lookup: track_id → _Track
        self._track_map: Dict[int, _Track] = {}
        self._id_gen = itertools.count(start=1)

        logger.info(
            "Tracker: gating=%.1fmm  max_missed=%d  min_confirm=%d",
            base_gating_mm, max_missed_frames, min_confirmations
        )

    # ============================================================
    # UPDATE
    # ============================================================

    def update(
        self,
        detections:            List[Detection],
        timestamp_ns:          int,
        encoder_velocity_mm_s: float,
        belt_position_mm:      float = 0.0
    ) -> List[TrackState]:

        tracks = list(self._track_map.values())

        # ---- 1. Predict all tracks ----
        for tr in tracks:
            dt = (timestamp_ns - tr.timestamp_ns) / 1e9
            tr.predict(dt)

        # ---- 2. Hungarian matching per class ----
        matched_det_indices:   List[int] = []
        matched_track_ids:     List[int] = []

        class_ids = set(d.class_id for d in detections)

        for cls in class_ids:

            # Direct index tracking — no .index() O(N) calls
            cls_det_indices   = [i for i, d in enumerate(detections) if d.class_id == cls]
            cls_track_ids     = [t.track_id for t in tracks if t.class_id == cls]

            if not cls_det_indices or not cls_track_ids:
                continue

            nd = len(cls_det_indices)
            nt = len(cls_track_ids)

            cost = np.full((nd, nt), fill_value=1e9, dtype=np.float64)

            for di, det_idx in enumerate(cls_det_indices):
                for ti, tid in enumerate(cls_track_ids):
                    dx = abs(detections[det_idx].x_mm - self._track_map[tid].x_mm)
                    if dx <= self._base_gating_mm:
                        cost[di, ti] = dx

            row_ind, col_ind = linear_sum_assignment(cost)

            for di, ti in zip(row_ind, col_ind):
                if cost[di, ti] >= 1e9:
                    continue
                matched_det_indices.append(cls_det_indices[di])
                matched_track_ids.append(cls_track_ids[ti])

        matched_track_id_set = set(matched_track_ids)

        # ---- 3. Update matched tracks ----
        for det_idx, tid in zip(matched_det_indices, matched_track_ids):
            det = detections[det_idx]
            tr  = self._track_map[tid]
            dt  = (det.timestamp_ns - tr.timestamp_ns) / 1e9
            tr.update(det, dt)
            tr.belt_position_mm = belt_position_mm
            tr.confirmed = (tr.age_frames >= self._min_confirmations)

        # ---- 4. New tracks for unmatched detections ----
        new_track_ids: set = set()
        for di, det in enumerate(detections):
            if di in matched_det_indices:
                continue
            new_id = next(self._id_gen)
            new_track_ids.add(new_id)
            tr = _Track(
                track_id          = new_id,
                detection         = det,
                process_noise     = self._process_noise,
                measurement_noise = self._measurement_noise
            )
            if abs(encoder_velocity_mm_s) > 1.0:
                tr.x_kf[1] = encoder_velocity_mm_s
            tr.belt_position_mm = belt_position_mm
            tr.confirmed = (tr.age_frames >= self._min_confirmations)
            self._track_map[new_id] = tr

        # ---- 5. Increment missed frames ----
        for tid, tr in self._track_map.items():
            if tid not in matched_track_id_set and tid not in new_track_ids:
                tr.missed_frames += 1

        # ---- 6. Prune dead tracks ----
        dead = [
            tid for tid, tr in self._track_map.items()
            if tr.missed_frames > self._max_missed_frames
        ]
        for tid in dead:
            del self._track_map[tid]

        if dead:
            logger.debug("Pruned %d dead tracks", len(dead))

        # ---- 7. Return confirmed states ----
        states = []
        for tr in self._track_map.values():
            if not tr.confirmed:
                continue
            states.append(TrackState(
                object_id        = tr.track_id,
                x_mm             = tr.x_mm,
                y_mm             = tr.y_mm,
                velocity_mm_s    = tr.velocity_mm_s,
                class_id         = tr.class_id,
                area_mm2         = tr.area_mm2,
                timestamp_ns     = timestamp_ns,
                belt_position_mm = tr.belt_position_mm,
                age_frames       = tr.age_frames,
                confirmed        = tr.confirmed
            ))

        return states

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> None:
        count = len(self._track_map)
        self._track_map.clear()
        logger.warning("Tracker reset: cleared %d tracks", count)
