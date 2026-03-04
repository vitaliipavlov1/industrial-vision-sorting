# core/interfaces.py

from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import Frame, Detection, TrackState, FireEvent, SystemFault


# ============================================================
# CAMERA INTERFACE
# ============================================================

class CameraInterface(ABC):
    """
    Abstract camera interface.

    Implementations:
        - USB camera (OpenCV)
        - GigE camera (Aravis / Harvesters)
        - Industrial camera SDK (Basler, FLIR, IDS)
        - Simulation / replay source
    """

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def grab_frame(self) -> Optional[object]:
        """
        Returns raw image (numpy HWC uint8) or None on timeout.
        Must be non-blocking with internal timeout.
        """
        pass


# ============================================================
# ENCODER INTERFACE
# ============================================================

class EncoderInterface(ABC):
    """
    Abstract belt encoder interface.

    Used to correlate object position with belt movement,
    enabling accurate fire timing independent of belt speed variation.
    """

    @abstractmethod
    def get_position_mm(self) -> float:
        """Returns absolute belt position in mm."""
        pass

    @abstractmethod
    def get_velocity_mm_s(self) -> float:
        """Returns current belt velocity in mm/s."""
        pass


# ============================================================
# SEGMENTATION ENGINE INTERFACE
# ============================================================

class SegmentationEngineInterface(ABC):
    """
    Interface for neural segmentation backend.
    """

    @abstractmethod
    def infer(self, image: object) -> int:
        """
        Runs inference on image.

        Returns:
            device pointer (int) to probability maps on GPU.
        """
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """
        Blocks until all GPU operations on this engine's stream are complete.
        Must be called before accessing GPU outputs from CPU.
        """
        pass

    @abstractmethod
    def destroy(self) -> None:
        pass


# ============================================================
# CLASSICAL CV PIPELINE INTERFACE
# ============================================================

class ClassicalCVInterface(ABC):
    """
    Interface for classical computer vision processing.
    Runs on CPU after GPU→CPU transfer.
    """

    @abstractmethod
    def process(self, class_map, timestamp_ns: int) -> List[Detection]:
        pass


# ============================================================
# TRACKER INTERFACE
# ============================================================

class TrackerInterface(ABC):

    @abstractmethod
    def update(
        self,
        detections:              List[Detection],
        timestamp_ns:            int,
        encoder_velocity_mm_s:   float,
        belt_position_mm:        float
    ) -> List[TrackState]:
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clears all active tracks. Called on fault recovery."""
        pass


# ============================================================
# PREDICTOR INTERFACE
# ============================================================

class PredictorInterface(ABC):

    @abstractmethod
    def compute_fire_event(
        self,
        track:           TrackState,
        current_time_ns: int
    ) -> Optional[FireEvent]:
        pass


# ============================================================
# SCHEDULER INTERFACE
# ============================================================

class SchedulerInterface(ABC):

    @abstractmethod
    def schedule(self, event: FireEvent) -> None:
        pass

    @abstractmethod
    def cancel_object(self, object_id: int) -> int:
        """
        Cancels all pending events for a given object_id.
        Returns number of events cancelled.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass


# ============================================================
# EJECTOR INTERFACE
# ============================================================

class EjectorInterface(ABC):

    @abstractmethod
    def fire(self, duration_ms: int) -> None:
        pass

    @abstractmethod
    def disable(self) -> None:
        """
        Emergency stop. Sets pin LOW and prevents further firing.
        Thread-safe.
        """
        pass

    @abstractmethod
    def enable(self) -> None:
        """
        Re-enables ejector after fault clearance.
        Should only be called by authorized fault recovery logic.
        """
        pass

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        pass


# ============================================================
# FAULT HANDLER INTERFACE
# ============================================================

class FaultHandlerInterface(ABC):
    """
    Centralized fault reporting and handling.

    Implementations may log to file, send MQTT alert,
    write to PLC register, send email, etc.
    """

    @abstractmethod
    def report(self, fault: SystemFault) -> None:
        pass

    @abstractmethod
    def get_history(self) -> List[SystemFault]:
        pass
