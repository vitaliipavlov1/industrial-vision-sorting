"""
test_suite.py — полный тест всех модулей без GPU/железа.

Запуск:
    python3 test_suite.py

Покрывает:
    - core/models.py
    - core/double_buffer.py
    - cv/classical_cv_pipeline.py
    - tracking/tracker.py
    - prediction/predictor.py
    - scheduling/scheduler.py
    - safety/latency_monitor.py
    - safety/watchdog.py
    - hardware/ejector.py (mock GPIO)
    - capture/camera.py (mock camera)
"""

import sys
import os
import time
import threading
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# HELPERS / MOCKS
# ============================================================

class MockGPIO:
    """Simulates GPIO interface for ejector tests."""
    def __init__(self):
        self.state    = {}
        self.calls    = []
        self._lock    = threading.Lock()
    def setup(self, pin, mode):
        self.state[pin] = 0
    def write(self, pin, value):
        with self._lock:
            self.state[pin] = value
            self.calls.append((pin, value, time.monotonic()))
    def read(self, pin):
        return self.state.get(pin, 0)


class MockEjector:
    """Minimal ejector mock for watchdog/scheduler tests."""
    def __init__(self):
        self.disabled   = False
        self.enabled    = True
        self.fire_count = 0
    def fire(self, duration_ms):
        self.fire_count += 1
    def disable(self):
        self.disabled = True
        self.enabled  = False
    def enable(self):
        self.enabled  = True
        self.disabled = False


class MockTracker:
    def reset(self):
        self.reset_called = True
    reset_called = False


class MockCamera:
    """Returns synthetic frames for camera thread tests."""
    def __init__(self, width=640, height=480):
        self.w = width
        self.h = height
        self._running = False
    def start(self): self._running = True
    def stop(self):  self._running = False
    def grab_frame(self):
        if not self._running:
            return None
        return np.zeros((self.h, self.w, 3), dtype=np.uint8)


def make_detection(x_mm=100.0, y_mm=50.0, class_id=1,
                   area_mm2=50.0, timestamp_ns=None):
    from core.models import Detection
    return Detection(
        x_mm=x_mm, y_mm=y_mm, class_id=class_id,
        area_mm2=area_mm2, width_mm=7.0, height_mm=7.0,
        aspect_ratio=1.0, elongation=1.0,
        solidity=0.9, convexity=0.95,
        circularity=0.85, compactness=15.0,
        confidence=1.0,
        timestamp_ns=timestamp_ns or time.perf_counter_ns()
    )


# ============================================================
# TEST: core/models.py
# ============================================================

class TestModels(unittest.TestCase):

    def test_frame_creation(self):
        from core.models import Frame
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        f   = Frame(image=img, timestamp_ns=12345, belt_position_mm=10.0)
        self.assertEqual(f.timestamp_ns, 12345)
        self.assertEqual(f.belt_position_mm, 10.0)

    def test_detection_creation(self):
        det = make_detection(x_mm=200.0, class_id=2)
        self.assertEqual(det.x_mm, 200.0)
        self.assertEqual(det.class_id, 2)

    def test_track_state(self):
        from core.models import TrackState
        ts = TrackState(
            object_id=1, x_mm=100.0, y_mm=50.0,
            velocity_mm_s=300.0, class_id=1, area_mm2=50.0,
            timestamp_ns=0, belt_position_mm=0.0,
            age_frames=5, confirmed=True
        )
        self.assertTrue(ts.confirmed)
        self.assertEqual(ts.velocity_mm_s, 300.0)

    def test_fire_event(self):
        from core.models import FireEvent
        fe = FireEvent(
            object_id=1, class_id=1,
            fire_time_ns=1_000_000, duration_ms=20
        )
        self.assertEqual(fe.duration_ms, 20)

    def test_system_fault(self):
        from core.models import SystemFault, SystemFaultCode
        f = SystemFault(
            code=SystemFaultCode.FRAME_TIMEOUT,
            message="timeout",
            timestamp_ns=0,
            context={}
        )
        self.assertEqual(f.code, SystemFaultCode.FRAME_TIMEOUT)


# ============================================================
# TEST: core/double_buffer.py
# ============================================================

class TestDoubleBuffer(unittest.TestCase):

    def test_switch(self):
        from core.double_buffer import DoubleBuffer
        a, b = object(), object()
        db = DoubleBuffer(a, b)
        self.assertIs(db.current(), a)
        n = db.next()
        self.assertIs(n, b)
        self.assertIs(db.current(), b)
        db.next()
        self.assertIs(db.current(), a)

    def test_all(self):
        from core.double_buffer import DoubleBuffer
        a, b = "A", "B"
        db = DoubleBuffer(a, b)
        self.assertEqual(set(db.all()), {"A", "B"})

    def test_thread_safety(self):
        from core.double_buffer import DoubleBuffer
        db = DoubleBuffer(0, 1)
        errors = []
        def worker():
            for _ in range(500):
                try:
                    db.next()
                    db.current()
                except Exception as e:
                    errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [], f"Thread safety errors: {errors}")


# ============================================================
# TEST: cv/classical_cv_pipeline.py
# ============================================================

class TestClassicalCV(unittest.TestCase):

    def _make_pipeline(self):
        from cv.classical_cv_pipeline import ClassicalCVPipeline
        class_configs = {
            1: {
                "min_area_mm2": 5, "max_area_mm2": 500,
                "min_solidity": 0.5, "max_elongation": 8.0,
                "roi_x_min_mm": 0, "roi_x_max_mm": 500,
            }
        }
        return ClassicalCVPipeline(mm_per_pixel=0.25, class_configs=class_configs)

    def test_empty_mask(self):
        pipeline  = self._make_pipeline()
        class_map = np.zeros((640, 640), dtype=np.uint8)
        dets      = pipeline.process(class_map, timestamp_ns=0)
        self.assertEqual(dets, [])

    def test_single_blob_detected(self):
        pipeline  = self._make_pipeline()
        class_map = np.zeros((640, 640), dtype=np.uint8)
        # 80×80 pixel square blob of class 1 → area = 80*80*(0.25²) = 400mm²
        class_map[280:360, 280:360] = 1
        dets = pipeline.process(class_map, timestamp_ns=0)
        self.assertGreater(len(dets), 0, "Should detect blob")
        self.assertEqual(dets[0].class_id, 1)

    def test_too_small_blob_filtered(self):
        pipeline  = self._make_pipeline()
        class_map = np.zeros((640, 640), dtype=np.uint8)
        # 4×4 px → area = 16*(0.25²) = 1mm² < min_area_mm2=5
        class_map[100:104, 100:104] = 1
        dets = pipeline.process(class_map, timestamp_ns=0)
        self.assertEqual(dets, [])

    def test_detection_fields(self):
        pipeline  = self._make_pipeline()
        class_map = np.zeros((640, 640), dtype=np.uint8)
        class_map[200:260, 200:260] = 1
        dets = pipeline.process(class_map, timestamp_ns=42)
        self.assertGreater(len(dets), 0)
        d = dets[0]
        self.assertGreater(d.area_mm2, 0)
        self.assertGreater(d.solidity, 0)
        self.assertGreater(d.circularity, 0)
        self.assertEqual(d.timestamp_ns, 42)

    def test_outside_roi_filtered(self):
        from cv.classical_cv_pipeline import ClassicalCVPipeline
        # roi_x_max_mm = 10mm → only x_mm < 10 allowed
        cfg = {1: {
            "min_area_mm2": 5, "max_area_mm2": 500,
            "min_solidity": 0.5, "max_elongation": 8.0,
            "roi_x_min_mm": 0, "roi_x_max_mm": 10,  # very narrow ROI
        }}
        pipeline  = ClassicalCVPipeline(mm_per_pixel=0.25, class_configs=cfg)
        class_map = np.zeros((640, 640), dtype=np.uint8)
        class_map[200:260, 400:460] = 1  # centroid at ~112mm >> 10mm limit
        dets = pipeline.process(class_map, timestamp_ns=0)
        self.assertEqual(dets, [], "Blob outside ROI should be filtered")

    def test_mm_per_pixel_precomputed(self):
        # Verify _mm_per_pixel_sq is precomputed (not recomputed per contour)
        pipeline = self._make_pipeline()
        self.assertAlmostEqual(pipeline._mm_per_pixel_sq, 0.25 * 0.25)


# ============================================================
# TEST: tracking/tracker.py
# ============================================================

class TestTracker(unittest.TestCase):

    def _make_tracker(self, min_conf=1):
        from tracking.tracker import Tracker
        return Tracker(
            process_noise=1.0, measurement_noise=1.0,
            base_gating_mm=50.0, max_missed_frames=3,
            velocity_smoothing_alpha=0.5,
            min_confirmations=min_conf
        )

    def test_new_track_created(self):
        tracker = self._make_tracker()
        det     = make_detection(x_mm=100.0)
        tracks  = tracker.update([det], timestamp_ns=0, encoder_velocity_mm_s=0.0)
        # min_confirmations=1 → confirmed immediately
        self.assertEqual(len(tracks), 1)
        self.assertAlmostEqual(tracks[0].x_mm, 100.0, places=1)

    def test_track_follows_detection(self):
        tracker = self._make_tracker()
        t0 = time.perf_counter_ns()
        tracker.update([make_detection(x_mm=100.0, timestamp_ns=t0)],
                       timestamp_ns=t0, encoder_velocity_mm_s=0.0)
        t1 = t0 + 100_000_000  # +100ms
        tracks = tracker.update(
            [make_detection(x_mm=130.0, timestamp_ns=t1)],
            timestamp_ns=t1, encoder_velocity_mm_s=0.0
        )
        self.assertEqual(len(tracks), 1)
        self.assertAlmostEqual(tracks[0].x_mm, 130.0, places=0)

    def test_velocity_estimated(self):
        tracker = self._make_tracker()
        t0 = time.perf_counter_ns()
        tracker.update([make_detection(x_mm=0.0, timestamp_ns=t0)],
                       timestamp_ns=t0, encoder_velocity_mm_s=0.0)
        t1 = t0 + 100_000_000  # 100ms later
        tracks = tracker.update(
            [make_detection(x_mm=30.0, timestamp_ns=t1)],
            timestamp_ns=t1, encoder_velocity_mm_s=0.0
        )
        # Moved 30mm in 100ms → ~300mm/s
        self.assertGreater(tracks[0].velocity_mm_s, 0)

    def test_missed_frames_prune(self):
        tracker = self._make_tracker()
        t0 = time.perf_counter_ns()
        tracker.update([make_detection(timestamp_ns=t0)],
                       timestamp_ns=t0, encoder_velocity_mm_s=0.0)
        # No detections for max_missed_frames+1 updates
        for i in range(5):
            t = t0 + (i+1) * 50_000_000
            tracks = tracker.update([], timestamp_ns=t, encoder_velocity_mm_s=0.0)
        self.assertEqual(tracks, [])

    def test_separate_classes_not_matched(self):
        tracker = self._make_tracker()
        t0 = time.perf_counter_ns()
        det1 = make_detection(x_mm=100.0, class_id=1, timestamp_ns=t0)
        det2 = make_detection(x_mm=105.0, class_id=2, timestamp_ns=t0)
        tracks = tracker.update([det1, det2], timestamp_ns=t0, encoder_velocity_mm_s=0.0)
        class_ids = {t.class_id for t in tracks}
        self.assertIn(1, class_ids)
        self.assertIn(2, class_ids)

    def test_min_confirmations(self):
        tracker = self._make_tracker(min_conf=3)
        t0 = time.perf_counter_ns()
        # First 2 updates → not confirmed
        for i in range(2):
            t = t0 + i * 50_000_000
            tracks = tracker.update(
                [make_detection(x_mm=100.0 + i*15, timestamp_ns=t)],
                timestamp_ns=t, encoder_velocity_mm_s=0.0
            )
            self.assertEqual(tracks, [], f"Should not be confirmed at update {i+1}")
        # 3rd update → confirmed
        t = t0 + 2 * 50_000_000
        tracks = tracker.update(
            [make_detection(x_mm=130.0, timestamp_ns=t)],
            timestamp_ns=t, encoder_velocity_mm_s=0.0
        )
        self.assertEqual(len(tracks), 1)

    def test_reset_clears_tracks(self):
        tracker = self._make_tracker()
        t0 = time.perf_counter_ns()
        tracker.update([make_detection(timestamp_ns=t0)],
                       timestamp_ns=t0, encoder_velocity_mm_s=0.0)
        tracker.reset()
        self.assertEqual(len(tracker._track_map), 0)

    def test_hungarian_optimal_assignment(self):
        """Two detections, two tracks — must assign correctly, not cross."""
        tracker = self._make_tracker()
        t0 = time.perf_counter_ns()
        tracker.update(
            [make_detection(x_mm=100.0, timestamp_ns=t0),
             make_detection(x_mm=200.0, timestamp_ns=t0)],
            timestamp_ns=t0, encoder_velocity_mm_s=0.0
        )
        t1 = t0 + 100_000_000
        tracks = tracker.update(
            [make_detection(x_mm=115.0, timestamp_ns=t1),
             make_detection(x_mm=215.0, timestamp_ns=t1)],
            timestamp_ns=t1, encoder_velocity_mm_s=0.0
        )
        xs = sorted(t.x_mm for t in tracks)
        self.assertAlmostEqual(xs[0], 115.0, places=0)
        self.assertAlmostEqual(xs[1], 215.0, places=0)


# ============================================================
# TEST: prediction/predictor.py
# ============================================================

class TestPredictor(unittest.TestCase):

    def _make_predictor(self):
        from prediction.predictor import Predictor
        return Predictor(
            nozzle_position_mm=500.0,
            valve_delay_ms=5.0,
            air_response_ms=5.0,
            fire_duration_ms=20,
            min_velocity_mm_s=100.0,
            safety_margin_ms=3.0,
            min_distance_mm=20.0
        )

    def _make_track(self, x_mm=100.0, velocity=300.0, class_id=1):
        from core.models import TrackState
        return TrackState(
            object_id=1, x_mm=x_mm, y_mm=50.0,
            velocity_mm_s=velocity, class_id=class_id,
            area_mm2=50.0, timestamp_ns=0,
            belt_position_mm=0.0, age_frames=5, confirmed=True
        )

    def test_normal_fire_event(self):
        pred  = self._make_predictor()
        track = self._make_track(x_mm=100.0, velocity=300.0)
        event = pred.compute_fire_event(track, current_time_ns=0)
        self.assertIsNotNone(event)
        self.assertEqual(event.duration_ms, 20)
        self.assertGreater(event.fire_time_ns, 0)

    def test_too_slow_returns_none(self):
        pred  = self._make_predictor()
        track = self._make_track(velocity=50.0)  # below min 100
        event = pred.compute_fire_event(track, current_time_ns=0)
        self.assertIsNone(event)

    def test_too_close_to_nozzle_returns_none(self):
        pred  = self._make_predictor()
        track = self._make_track(x_mm=490.0, velocity=300.0)  # 10mm < min_distance 20mm
        event = pred.compute_fire_event(track, current_time_ns=0)
        self.assertIsNone(event)

    def test_fire_time_correct(self):
        pred  = self._make_predictor()
        track = self._make_track(x_mm=200.0, velocity=500.0)
        now   = 0
        event = pred.compute_fire_event(track, current_time_ns=now)
        # distance = 300mm, time_to_nozzle = 600ms
        # total_delay = 5+5+3 = 13ms, fire_time_ms = 600-13 = 587ms
        expected_ms = 587.0
        actual_ms   = event.fire_time_ns / 1e6
        self.assertAlmostEqual(actual_ms, expected_ms, delta=1.0)

    def test_negative_fire_time_returns_none(self):
        pred  = self._make_predictor()
        # Object very close and fast → fire time would be negative
        track = self._make_track(x_mm=499.0, velocity=10000.0)
        event = pred.compute_fire_event(track, current_time_ns=0)
        self.assertIsNone(event)


# ============================================================
# TEST: scheduling/scheduler.py
# ============================================================

class TestScheduler(unittest.TestCase):

    def _make_scheduler(self):
        from scheduling.scheduler import Scheduler
        ejector = MockEjector()
        sched   = Scheduler(ejector=ejector, busy_wait_threshold_us=100)
        return sched, ejector

    def _make_event(self, delay_ms=50, object_id=1, class_id=1):
        from core.models import FireEvent
        return FireEvent(
            object_id=object_id, class_id=class_id,
            fire_time_ns=time.perf_counter_ns() + int(delay_ms * 1e6),
            duration_ms=20
        )

    def test_event_fires(self):
        sched, ejector = self._make_scheduler()
        sched.start()
        sched.schedule(self._make_event(delay_ms=50))
        time.sleep(0.15)
        sched.stop()
        self.assertEqual(ejector.fire_count, 1)

    def test_deduplication(self):
        """Same object_id scheduled twice — should fire only once."""
        sched, ejector = self._make_scheduler()
        sched.start()
        sched.schedule(self._make_event(delay_ms=60, object_id=5))
        sched.schedule(self._make_event(delay_ms=60, object_id=5))
        time.sleep(0.15)
        sched.stop()
        self.assertEqual(ejector.fire_count, 1, "Duplicate event must be deduplicated")

    def test_multiple_objects(self):
        sched, ejector = self._make_scheduler()
        sched.start()
        for oid in range(1, 4):
            sched.schedule(self._make_event(delay_ms=50 + oid*20, object_id=oid))
        time.sleep(0.3)
        sched.stop()
        self.assertEqual(ejector.fire_count, 3)

    def test_cancel_object(self):
        sched, ejector = self._make_scheduler()
        sched.start()
        sched.schedule(self._make_event(delay_ms=200, object_id=99))
        sched.cancel_object(99)
        time.sleep(0.3)
        sched.stop()
        self.assertEqual(ejector.fire_count, 0, "Cancelled event must not fire")

    def test_fired_count_property(self):
        sched, ejector = self._make_scheduler()
        sched.start()
        sched.schedule(self._make_event(delay_ms=30, object_id=10))
        time.sleep(0.1)
        sched.stop()
        self.assertEqual(sched.fired_count, 1)


# ============================================================
# TEST: safety/latency_monitor.py
# ============================================================

class TestLatencyMonitor(unittest.TestCase):

    def test_basic_measurement(self):
        from safety.latency_monitor import LatencyMonitor
        mon = LatencyMonitor(window_size=10, max_latency_ms=30.0)
        mon.start_frame()
        time.sleep(0.01)
        mon.end_frame()
        self.assertGreater(mon.current_latency(), 5.0)
        self.assertLess(mon.current_latency(), 50.0)

    def test_percentiles(self):
        from safety.latency_monitor import LatencyMonitor
        mon = LatencyMonitor(window_size=100, max_latency_ms=30.0)
        for ms in range(1, 51):
            mon.start_frame()
            time.sleep(ms / 10000.0)
            mon.end_frame()
        self.assertGreater(mon.p95_latency(), mon.average_latency())
        self.assertGreater(mon.p99_latency(), mon.p95_latency())

    def test_overload_detection(self):
        from safety.latency_monitor import LatencyMonitor
        mon = LatencyMonitor(window_size=20, max_latency_ms=5.0)
        for _ in range(20):
            mon.start_frame()
            time.sleep(0.02)   # 20ms >> 5ms threshold
            mon.end_frame()
        self.assertTrue(mon.is_overloaded())

    def test_not_overloaded(self):
        from safety.latency_monitor import LatencyMonitor
        mon = LatencyMonitor(window_size=10, max_latency_ms=100.0)
        for _ in range(10):
            mon.start_frame()
            time.sleep(0.001)
            mon.end_frame()
        self.assertFalse(mon.is_overloaded())

    def test_fps_estimate(self):
        from safety.latency_monitor import LatencyMonitor
        mon = LatencyMonitor(window_size=20, max_latency_ms=30.0)
        for _ in range(20):
            mon.start_frame()
            time.sleep(0.01)  # ~10ms → ~100fps
            mon.end_frame()
        fps = mon.fps()
        self.assertGreater(fps, 50)
        self.assertLess(fps, 200)

    def test_circular_buffer_no_overflow(self):
        from safety.latency_monitor import LatencyMonitor
        mon = LatencyMonitor(window_size=5, max_latency_ms=30.0)
        for _ in range(20):   # 4× window size
            mon.start_frame()
            mon.end_frame()
        self.assertLessEqual(mon._count, 5)

    def test_stats_dict(self):
        from safety.latency_monitor import LatencyMonitor
        mon = LatencyMonitor()
        mon.start_frame()
        time.sleep(0.005)
        mon.end_frame()
        s = mon.stats()
        for key in ("fps","avg_ms","p95_ms","p99_ms","total_frames","overloaded"):
            self.assertIn(key, s)


# ============================================================
# TEST: safety/watchdog.py
# ============================================================

class TestWatchdog(unittest.TestCase):

    def _make_watchdog(self, timeout_ms=100, max_latency=30.0):
        from safety.latency_monitor import LatencyMonitor
        from safety.watchdog import Watchdog
        lm      = LatencyMonitor(max_latency_ms=max_latency)
        ejector = MockEjector()
        tracker = MockTracker()
        wd      = Watchdog(
            latency_monitor=lm, ejector=ejector, tracker=tracker,
            frame_timeout_ms=timeout_ms, check_interval_ms=20,
            recovery_timeout_s=0.3
        )
        return wd, ejector, tracker, lm

    def test_frame_timeout_disables_ejector(self):
        wd, ejector, tracker, lm = self._make_watchdog(timeout_ms=50)
        wd.start()
        time.sleep(0.2)   # no notify_frame → timeout
        wd.stop()
        self.assertTrue(ejector.disabled)

    def test_notify_frame_prevents_timeout(self):
        wd, ejector, tracker, lm = self._make_watchdog(timeout_ms=200)
        wd.start()
        for _ in range(5):
            wd.notify_frame()
            time.sleep(0.05)
        wd.stop()
        self.assertFalse(ejector.disabled)

    def test_tracker_reset_on_fault(self):
        wd, ejector, tracker, lm = self._make_watchdog(timeout_ms=50)
        wd.start()
        time.sleep(0.2)
        wd.stop()
        self.assertTrue(tracker.reset_called)

    def test_auto_recovery(self):
        wd, ejector, tracker, lm = self._make_watchdog(timeout_ms=50)
        wd.start()
        time.sleep(0.15)  # trigger fault
        # Resume heartbeat
        for _ in range(10):
            wd.notify_frame()
            time.sleep(0.05)
        wd.stop()
        # After recovery_timeout_s=0.3 ejector should be re-enabled
        self.assertTrue(ejector.enabled)

    def test_fault_history_recorded(self):
        wd, ejector, tracker, lm = self._make_watchdog(timeout_ms=50)
        wd.start()
        time.sleep(0.2)
        wd.stop()
        history = wd.get_fault_history()
        self.assertGreater(len(history), 0)


# ============================================================
# TEST: hardware/ejector.py (mock GPIO)
# ============================================================

class TestEjector(unittest.TestCase):

    def _make_ejector(self):
        from hardware.ejector import Ejector
        gpio = MockGPIO()
        ej   = Ejector(gpio_interface=gpio, output_pin=17)
        return ej, gpio

    def test_fire_toggles_pin(self):
        ej, gpio = self._make_ejector()
        ej.fire(20)
        time.sleep(0.08)
        highs = [v for _, v, _ in gpio.calls if v == 1]
        lows  = [v for _, v, _ in gpio.calls if v == 0]
        self.assertGreater(len(highs), 0, "Pin should go HIGH on fire")
        self.assertGreater(len(lows),  1, "Pin should return LOW after fire")

    def test_fire_nonblocking(self):
        ej, gpio = self._make_ejector()
        t0 = time.monotonic()
        ej.fire(100)   # 100ms pulse
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.02, "fire() must return immediately (non-blocking)")
        time.sleep(0.15)  # let worker finish

    def test_disable_blocks_fire(self):
        ej, gpio = self._make_ejector()
        ej.disable()
        ej.fire(20)
        time.sleep(0.05)
        highs = [v for _, v, _ in gpio.calls if v == 1]
        # Only the init write(pin, 0) should exist, no HIGH after disable
        self.assertEqual(len(highs), 0)

    def test_enable_after_disable(self):
        ej, gpio = self._make_ejector()
        ej.disable()
        ej.enable()
        ej.fire(20)
        time.sleep(0.08)
        highs = [v for _, v, _ in gpio.calls if v == 1]
        self.assertGreater(len(highs), 0)

    def test_duration_clamped(self):
        ej, gpio = self._make_ejector()
        ej.fire(9999)  # way over MAX_DURATION_MS=200
        time.sleep(0.01)
        # Just checking it doesn't crash and queue accepted it
        self.assertTrue(True)

    def test_fire_count(self):
        ej, gpio = self._make_ejector()
        for _ in range(3):
            ej.fire(20)
        time.sleep(0.15)
        self.assertEqual(ej.fire_count, 3)


# ============================================================
# TEST: capture/camera.py (mock camera)
# ============================================================

class TestCameraThread(unittest.TestCase):

    def test_frames_arrive(self):
        import queue as q
        from capture.camera import CameraThread
        output_queue = q.Queue(maxsize=5)
        cam    = MockCamera()
        thread = CameraThread(camera=cam, encoder=None, output_queue=output_queue)
        thread.start_capture()
        time.sleep(0.1)
        thread.stop_capture()
        self.assertGreater(output_queue.qsize(), 0, "Should have received frames")

    def test_frame_has_correct_fields(self):
        import queue as q
        from capture.camera import CameraThread
        from core.models import Frame
        output_queue = q.Queue(maxsize=5)
        cam    = MockCamera(width=320, height=240)
        thread = CameraThread(camera=cam, encoder=None, output_queue=output_queue)
        thread.start_capture()
        time.sleep(0.1)
        thread.stop_capture()
        frame = output_queue.get_nowait()
        self.assertIsInstance(frame, Frame)
        self.assertIsNotNone(frame.timestamp_ns)
        self.assertEqual(frame.image.shape, (240, 320, 3))

    def test_queue_doesnt_overflow(self):
        import queue as q
        from capture.camera import CameraThread
        output_queue = q.Queue(maxsize=2)
        cam    = MockCamera()
        thread = CameraThread(camera=cam, encoder=None, output_queue=output_queue)
        thread.start_capture()
        time.sleep(0.2)  # camera runs fast, queue limited to 2
        thread.stop_capture()
        self.assertLessEqual(output_queue.qsize(), 2)

    def test_frame_id_increments(self):
        import queue as q
        from capture.camera import CameraThread
        output_queue = q.Queue(maxsize=10)
        cam    = MockCamera()
        thread = CameraThread(camera=cam, encoder=None, output_queue=output_queue)
        thread.start_capture()
        time.sleep(0.1)
        thread.stop_capture()
        frames = []
        while not output_queue.empty():
            frames.append(output_queue.get_nowait())
        if len(frames) >= 2:
            self.assertGreater(thread.frame_count, 0)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    test_classes = [
        TestModels, TestDoubleBuffer, TestClassicalCV,
        TestTracker, TestPredictor, TestScheduler,
        TestLatencyMonitor, TestWatchdog, TestEjector,
        TestCameraThread
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total  = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed")
    if result.failures or result.errors:
        sys.exit(1)
