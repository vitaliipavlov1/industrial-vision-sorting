# core/models.py

from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


# ============================================================
# ENUMS
# ============================================================

class SystemFaultCode(IntEnum):
    """
    Fault codes for structured fault reporting.
    """
    NONE              = 0
    FRAME_TIMEOUT     = 1
    LATENCY_OVERLOAD  = 2
    GPU_SYNC_ERROR    = 3
    EJECTOR_FAULT     = 4
    CAMERA_FAULT      = 5
    WATCHDOG_TRIGGER  = 6


# ============================================================
# FRAME STRUCTURE
# ============================================================

@dataclass
class Frame:
    """
    Frame captured from camera.

    Fields:
        image            - raw image array (numpy HWC uint8)
        timestamp_ns     - capture time (perf_counter_ns)
        belt_position_mm - encoder position at capture time (mm)
        frame_id         - monotonic frame counter
    """

    image:             object
    timestamp_ns:      int
    belt_position_mm:  float
    frame_id:          int = 0


# ============================================================
# DETECTION
# ============================================================

@dataclass
class Detection:
    """
    Object detection produced by classical CV layer.

    All spatial fields in mm (calibrated).
    """

    x_mm:          float
    y_mm:          float

    class_id:      int

    area_mm2:      float
    width_mm:      float
    height_mm:     float

    aspect_ratio:  float
    elongation:    float

    solidity:      float
    convexity:     float
    circularity:   float
    compactness:   float

    confidence:    float
    timestamp_ns:  int


# ============================================================
# TRACK STATE
# ============================================================

@dataclass
class TrackState:
    """
    Kalman-filtered track state.

    Fields:
        object_id        - unique track ID (monotonic)
        x_mm             - filtered X position on belt (mm)
        y_mm             - filtered Y position on belt (mm)
        velocity_mm_s    - estimated belt velocity (mm/s)
        class_id         - object class
        area_mm2         - smoothed detection area (mm²)
        timestamp_ns     - last update time (perf_counter_ns)
        belt_position_mm - belt encoder position at last update
        age_frames       - number of frames track has been alive
        confirmed        - True after min_confirmations frames
    """

    object_id:         int
    x_mm:              float
    y_mm:              float
    velocity_mm_s:     float
    class_id:          int
    area_mm2:          float
    timestamp_ns:      int
    belt_position_mm:  float  = 0.0
    age_frames:        int    = 0
    confirmed:         bool   = False


# ============================================================
# FIRE EVENT
# ============================================================

@dataclass
class FireEvent:
    """
    Scheduled ejector activation event.

    Fields:
        object_id    - track ID that triggered this event
        class_id     - object class
        fire_time_ns - absolute time to open valve (perf_counter_ns)
        duration_ms  - valve open duration (ms)
        sequence_id  - monotonic event counter for dedup
    """

    object_id:    int
    class_id:     int
    fire_time_ns: int
    duration_ms:  int
    # note: sequencing is managed by Scheduler internally (heap tuple key)


# ============================================================
# SYSTEM FAULT
# ============================================================

@dataclass
class SystemFault:
    """
    Structured fault record for logging and alerting.
    """

    code:        SystemFaultCode
    message:     str
    timestamp_ns: int
    context:     Optional[dict] = None
