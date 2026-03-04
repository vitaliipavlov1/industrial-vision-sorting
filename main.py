# main.py

import time
import logging
import logging.config
import signal
from queue import Queue, Empty

import yaml
import cupy as cp
import cupy.cuda

from capture.camera import CameraThread

from inference.segmentation_engine import SegmentationEngine
from inference.gpu_postprocess import GPUPostProcessor
from inference.gpu_roi_crop import GPURoiCrop

from cv.classical_cv_pipeline import ClassicalCVPipeline

from tracking.tracker import Tracker
from prediction.predictor import Predictor

from scheduling.scheduler import Scheduler
from hardware.ejector import Ejector

from safety.latency_monitor import LatencyMonitor
from safety.watchdog import Watchdog

from core.pinned_memory import PinnedMemory
from core.double_buffer import DoubleBuffer


# ============================================================
# LOGGING
# ============================================================

def _configure_logging() -> None:

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s.%(msecs)03d  %(levelname)-8s  "
                          "%(name)-30s  %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class":     "logging.StreamHandler",
                "formatter": "detailed",
                "stream":    "ext://sys.stdout"
            },
            "file": {
                "class":       "logging.handlers.RotatingFileHandler",
                "formatter":   "detailed",
                "filename":    "system.log",
                "maxBytes":    10 * 1024 * 1024,
                "backupCount": 5
            }
        },
        "root": {"level": "INFO", "handlers": ["console", "file"]}
    })


logger = logging.getLogger(__name__)


# ============================================================
# INDUSTRIAL VISION SYSTEM
# ============================================================

class IndustrialVisionSystem:
    """
    Real-time industrial vision sorting system.

    GPU pipeline (zero CPU round-trips, CuPy-backed):
    ┌──────────────────────────────────────────────────────────┐
    │  frame.image  (CPU pinned)                               │
    │      │  H2D [1 DMA, CuPy async, once per frame]         │
    │      ▼                                                   │
    │  _device_frame_buf  ──► roi_crop kernel  ──► device_roi  │
    │                                  │                       │
    │                         preprocess kernel                │
    │                         (bilinear resize+norm+CHW)       │
    │                                  │                       │
    │                         TRT execute_async_v2             │
    │                                  │                       │
    │                         postprocess kernel               │
    │                         (argmax + threshold)             │
    │                                  │  D2H async            │
    │                         host_class_map (pinned)          │
    └──────────────────────────────────────────────────────────┘
         ▼
    ClassicalCV (CPU) → Tracker → Predictor → Scheduler → Ejector

    CUDA stack:
        CuPy  — streams, device memory, pinned memory, custom kernels
        TensorRT (python) — neural inference (no CuPy equivalent)
    """

    def __init__(self, config: dict):

        self._config  = config
        self._running = False

        self._build_pipeline(config)
        self._register_signals()

        logger.info("IndustrialVisionSystem initialized")

    # ============================================================
    # BUILD PIPELINE
    # ============================================================

    def _build_pipeline(self, cfg: dict) -> None:

        seg_cfg  = cfg["segmentation"]
        roi_cfg  = cfg["roi"]
        trk_cfg  = cfg["tracking"]
        pred_cfg = cfg["prediction"]
        saf_cfg  = cfg["safety"]
        cam_cfg  = cfg["camera"]

        self._frame_queue = Queue(maxsize=2)
        self._encoder_obj = cfg.get("encoder_object")

        # ---- Camera ----
        self._camera_thread = CameraThread(
            camera       = cfg["camera_object"],
            encoder      = self._encoder_obj,
            output_queue = self._frame_queue
        )

        # ---- ROI dimensions (cached) ----
        self._roi_w = roi_cfg["width"]
        self._roi_h = roi_cfg["height"]

        # ---- ROI crop ----
        self._roi_crop = GPURoiCrop(
            roi_x = roi_cfg["x"],
            roi_y = roi_cfg["y"],
            roi_w = self._roi_w,
            roi_h = self._roi_h
        )

        # ---- Persistent device buffer for full frame upload ----
        frame_bytes = cam_cfg["width"] * cam_cfg["height"] * 3
        self._device_frame_buf = cp.cuda.alloc(frame_bytes)
        self._device_frame_bytes = frame_bytes

        logger.debug(
            "Allocated %.1f MB persistent frame device buffer",
            frame_bytes / (1024 * 1024)
        )

        # ---- Segmentation engine ----
        self._segmentation_engine = SegmentationEngine(
            engine_path  = seg_cfg["engine_path"],
            input_width  = seg_cfg["input_width"],
            input_height = seg_cfg["input_height"],
            num_classes  = seg_cfg["num_classes"]
        )

        # ---- GPU postprocessor ----
        self._postprocessor = GPUPostProcessor(
            width       = seg_cfg["input_width"],
            height      = seg_cfg["input_height"],
            num_classes = seg_cfg["num_classes"]
        )

        # ---- Pinned memory double-buffer (class map D2H) ----
        pixels = seg_cfg["input_width"] * seg_cfg["input_height"]
        self._class_map_buffers = DoubleBuffer(
            PinnedMemory(pixels),
            PinnedMemory(pixels)
        )
        self._class_map_shape = (seg_cfg["input_height"], seg_cfg["input_width"])

        # ---- Classical CV ----
        self._cv_pipeline = ClassicalCVPipeline(
            mm_per_pixel  = cfg["calibration"]["mm_per_pixel"],
            class_configs = cfg["classes"]
        )

        # ---- Tracker ----
        self._tracker = Tracker(
            process_noise            = trk_cfg["process_noise"],
            measurement_noise        = trk_cfg["measurement_noise"],
            base_gating_mm           = trk_cfg["base_gating_mm"],
            max_missed_frames        = trk_cfg["max_missed_frames"],
            velocity_smoothing_alpha = trk_cfg["velocity_smoothing_alpha"],
            min_confirmations        = trk_cfg.get("min_confirmations", 3)
        )

        # ---- Predictor ----
        self._predictor = Predictor(
            nozzle_position_mm        = pred_cfg["nozzle_position_mm"],
            valve_delay_ms            = pred_cfg["valve_delay_ms"],
            air_response_ms           = pred_cfg["air_response_ms"],
            fire_duration_ms          = pred_cfg["fire_duration_ms"],
            min_velocity_mm_s         = pred_cfg["min_velocity_mm_s"],
            safety_margin_ms          = pred_cfg["safety_margin_ms"],
            min_distance_mm           = pred_cfg["min_distance_mm"],
            encoder_velocity_fallback = pred_cfg.get("encoder_velocity_fallback", True),
            per_class_duration_ms     = pred_cfg.get("per_class_duration_ms")
        )

        # ---- Ejector ----
        self._ejector = Ejector(
            gpio_interface = cfg["gpio_interface"],
            output_pin     = cfg["ejector_pin"]
        )

        # ---- Scheduler ----
        self._scheduler = Scheduler(
            ejector                = self._ejector,
            busy_wait_threshold_us = cfg["scheduler"]["busy_wait_us"]
        )

        # ---- Latency monitor ----
        self._latency_monitor = LatencyMonitor(
            window_size    = 200,
            max_latency_ms = saf_cfg["max_latency_ms"]
        )

        # ---- Watchdog ----
        self._watchdog = Watchdog(
            latency_monitor    = self._latency_monitor,
            ejector            = self._ejector,
            tracker            = self._tracker,
            frame_timeout_ms   = saf_cfg["frame_timeout_ms"],
            recovery_timeout_s = saf_cfg.get("recovery_timeout_s", 5.0)
        )

        # ---- Nozzle position (cached) ----
        self._nozzle_position_mm = pred_cfg["nozzle_position_mm"]

        # ---- Scheduled objects: object_id → x_mm at schedule time ----
        self._scheduled_objects: dict = {}

    # ============================================================
    # SIGNAL HANDLING
    # ============================================================

    def _register_signals(self) -> None:
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        logger.info("Signal %d received — shutting down", signum)
        self._running = False

    # ============================================================
    # RUN
    # ============================================================

    def run(self) -> None:

        self._running = True

        self._camera_thread.start_capture()
        self._scheduler.start()
        self._watchdog.start()

        stream = self._segmentation_engine.stream

        logger.info("Pipeline running")

        stats_interval  = 5.0
        last_stats_time = time.monotonic()

        try:

            while self._running:

                # ------------------------------------------------
                # 1. ACQUIRE FRAME
                # ------------------------------------------------

                try:
                    frame = self._frame_queue.get(timeout=1.0)
                except Empty:
                    continue

                self._latency_monitor.start_frame()
                self._watchdog.notify_frame()

                # ------------------------------------------------
                # 2. ENCODER VELOCITY
                # ------------------------------------------------

                encoder_velocity_mm_s = 0.0

                if self._encoder_obj is not None:
                    try:
                        encoder_velocity_mm_s = self._encoder_obj.get_velocity_mm_s()
                    except Exception:
                        pass

                # ------------------------------------------------
                # 3. UPLOAD FRAME TO GPU  (1 H2D, persistent buffer)
                #    CuPy replaces cudaMalloc + cudaMemcpyAsync
                # ------------------------------------------------

                nbytes = frame.image.nbytes

                if nbytes > self._device_frame_bytes:
                    raise RuntimeError(
                        f"Frame {nbytes}B > device buffer {self._device_frame_bytes}B"
                    )

                cp.cuda.runtime.memcpyAsync(
                    self._device_frame_buf.ptr,
                    frame.image.ctypes.data,
                    nbytes,
                    cp.cuda.runtime.memcpyHostToDevice,
                    stream.ptr
                )

                # ------------------------------------------------
                # 4. ROI CROP  (GPU kernel, device→device)
                # ------------------------------------------------

                device_roi = self._roi_crop.execute(
                    device_input_ptr = self._device_frame_buf.ptr,
                    stream           = stream,
                    in_width         = frame.image.shape[1],
                    in_height        = frame.image.shape[0]
                )

                # ------------------------------------------------
                # 5. SEGMENTATION  (preprocess + TRT, device→device)
                # ------------------------------------------------

                device_probs = self._segmentation_engine.infer(
                    device_input_ptr = device_roi,
                    in_width         = self._roi_w,
                    in_height        = self._roi_h
                )

                # ------------------------------------------------
                # 6. POSTPROCESS  (argmax + threshold, device→device)
                # ------------------------------------------------

                self._postprocessor.execute(
                    device_probs,
                    stream,
                    self._config["segmentation"]["threshold"]
                )

                # ------------------------------------------------
                # 7. D2H COPY + SYNC
                #    CuPy replaces cudaMemcpyAsync + cudaStreamSynchronize
                # ------------------------------------------------

                buffer         = self._class_map_buffers.next()
                host_class_map = buffer.get_numpy(self._class_map_shape)

                self._postprocessor.copy_to_host(host_class_map, stream)

                # Single sync after all GPU work + D2H enqueue
                self._segmentation_engine.synchronize()

                # ------------------------------------------------
                # 8. CLASSICAL CV  (CPU)
                # ------------------------------------------------

                detections = self._cv_pipeline.process(
                    host_class_map,
                    frame.timestamp_ns
                )

                # ------------------------------------------------
                # 9. TRACKER
                # ------------------------------------------------

                tracks = self._tracker.update(
                    detections            = detections,
                    timestamp_ns          = frame.timestamp_ns,
                    encoder_velocity_mm_s = encoder_velocity_mm_s,
                    belt_position_mm      = frame.belt_position_mm
                )

                # ------------------------------------------------
                # 10. PREDICTION + SCHEDULING
                # ------------------------------------------------

                now_ns     = time.perf_counter_ns()
                active_ids = {tr.object_id for tr in tracks}

                gone = [oid for oid in self._scheduled_objects if oid not in active_ids]
                for oid in gone:
                    del self._scheduled_objects[oid]

                for tr in tracks:

                    if tr.object_id in self._scheduled_objects:
                        continue

                    if tr.x_mm >= self._nozzle_position_mm:
                        continue

                    event = self._predictor.compute_fire_event(
                        track                 = tr,
                        current_time_ns       = now_ns,
                        encoder_velocity_mm_s = encoder_velocity_mm_s
                    )

                    if event:
                        self._scheduler.schedule(event)
                        self._scheduled_objects[tr.object_id] = tr.x_mm

                # ------------------------------------------------
                # 11. LATENCY
                # ------------------------------------------------

                self._latency_monitor.end_frame()

                # ------------------------------------------------
                # 12. PERIODIC STATS
                # ------------------------------------------------

                now_mono = time.monotonic()

                if now_mono - last_stats_time >= stats_interval:

                    s = self._latency_monitor.stats()

                    logger.info(
                        "Stats | fps=%.1f  avg=%.1fms  p95=%.1fms  p99=%.1fms  "
                        "frames=%d  drops=%d  fires=%d  pending=%d  tracks=%d  faults=%d",
                        s["fps"], s["avg_ms"], s["p95_ms"], s["p99_ms"],
                        self._camera_thread.frame_count,
                        self._camera_thread.drop_count,
                        self._scheduler.fired_count,
                        self._scheduler.pending_count,
                        len(tracks),
                        len(self._watchdog.get_fault_history())
                    )

                    last_stats_time = now_mono

        except Exception as exc:
            logger.critical("Pipeline exception: %s", exc, exc_info=True)
            self._ejector.disable()
            raise

        finally:
            self._shutdown()

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def _shutdown(self) -> None:

        logger.info("Shutting down...")

        self._ejector.disable()

        self._camera_thread.stop_capture()
        self._scheduler.stop()
        self._watchdog.stop()

        self._segmentation_engine.destroy()
        self._postprocessor.destroy()
        self._roi_crop.destroy()

        # Release persistent frame buffer (CuPy GC handles it,
        # but explicit None ensures immediate release)
        self._device_frame_buf = None

        for buf in self._class_map_buffers.all():
            buf.free()

        logger.info(
            "Shutdown complete | frames=%d  drops=%d  fires=%d  faults=%d",
            self._camera_thread.frame_count,
            self._camera_thread.drop_count,
            self._scheduler.fired_count,
            len(self._watchdog.get_fault_history())
        )


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:

    _configure_logging()

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    cfg["camera_object"]  = None   # CameraInterface implementation
    cfg["encoder_object"] = None   # EncoderInterface implementation
    cfg["gpio_interface"] = None   # GPIO implementation
    cfg["ejector_pin"]    = cfg.get("ejector_pin", 17)

    system = IndustrialVisionSystem(cfg)
    system.run()


if __name__ == "__main__":
    main()
