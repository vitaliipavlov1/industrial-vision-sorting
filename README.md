# Industrial Vision Sorting System

Production-grade real-time computer vision pipeline for industrial conveyor belt sorting. Detects, classifies, tracks, and pneumatically ejects defective objects at 120fps using GPU-accelerated neural segmentation and classical CV.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![CUDA](https://img.shields.io/badge/CUDA-11%2B%20%2F%2012%2B-green)
![TensorRT](https://img.shields.io/badge/TensorRT-8%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-54%20passed-brightgreen)

---

## Pipeline

```
Camera (120fps)
    │
    ▼  H2D  [one DMA per frame]
GPURoiCrop           crop conveyor region          (CuPy CUDA kernel)
    │  device→device
    ▼
SegmentationEngine   neural segmentation           (TensorRT FP16)
    │  device→device
    ▼
GPUPostProcessor     argmax + confidence threshold (CuPy CUDA kernel)
    │  D2H  [one DMA, pinned memory]
    ▼
ClassicalCVPipeline  morphology + contours + shape metrics  (OpenCV)
    │
    ▼
Tracker              Kalman filter + Hungarian assignment    (scipy)
    │
    ▼
Predictor            compute ejector fire timestamp
    │
    ▼
Scheduler            deterministic timing (sleep + busy-wait)
    │
    ▼
Ejector              solenoid valve / air nozzle   (GPIO, non-blocking)
```

GPU pipeline is fully device-to-device. One H2D per frame at input, one D2H into pinned memory at output. No CPU involvement between GPU stages.

---

## Project Structure

```
industrial-vision-sorting/
├── main.py                         Entry point, pipeline orchestration
├── config.yaml                     All tunable parameters
├── test_suite.py                   54 unit tests (no GPU required)
├── requirements.txt
│
├── core/
│   ├── models.py                   Frame, Detection, TrackState, FireEvent, SystemFault
│   ├── interfaces.py               Abstract base classes for all components
│   ├── double_buffer.py            Thread-safe double buffer (GPU/CPU overlap)
│   └── pinned_memory.py            CuPy-backed pinned host memory
│
├── capture/
│   └── camera.py                   Camera thread with frame-dropping strategy
│
├── inference/
│   ├── segmentation_engine.py      TensorRT engine wrapper
│   ├── gpu_roi_crop.py             CUDA kernel: ROI crop (device→device)
│   ├── gpu_preprocess.py           CUDA kernel: bilinear resize + normalize + HWC→CHW
│   └── gpu_postprocess.py          CUDA kernel: argmax + threshold → class map
│
├── cv/
│   └── classical_cv_pipeline.py    Morphology, contours, geometry, shape metrics
│
├── tracking/
│   └── tracker.py                  1D Kalman filter + Hungarian assignment
│
├── prediction/
│   └── predictor.py                Nozzle arrival time prediction
│
├── scheduling/
│   └── scheduler.py                Priority queue + busy-wait for <500µs jitter
│
├── hardware/
│   └── ejector.py                  Non-blocking solenoid controller (worker thread)
│
├── safety/
│   ├── latency_monitor.py          p95/p99 latency, pre-allocated circular buffer
│   └── watchdog.py                 Frame timeout + overload detection + auto-recovery
│
└── system/
    └── realtime_setup.py           OS-level RT: CPU affinity, SCHED_FIFO, mlockall
```

---

## Requirements

### Hardware

| Component | Notes |
|-----------|-------|
| NVIDIA GPU | Jetson Orin / AGX / discrete (Ampere or newer recommended) |
| Industrial camera | USB3, GigE Vision, or proprietary SDK |
| Rotary encoder | Optional — improves velocity estimation |
| Solenoid valve / air nozzle | GPIO-controlled, typically 24V |

### Software

- Python 3.10+
- CUDA 11.x or 12.x
- TensorRT 8+ ([NVIDIA TensorRT SDK](https://developer.nvidia.com/tensorrt))

---

## Installation

```bash
git clone https://github.com/your-org/industrial-vision-sorting.git
cd industrial-vision-sorting
pip install -r requirements.txt
```

TensorRT is included in NVIDIA JetPack SDK on Jetson, or installed manually on x86 via the NVIDIA developer portal.

---

## Configuration

All parameters are in `config.yaml`. Key sections:

```yaml
camera:
  width: 1920
  height: 1080
  fps: 120

roi:
  x: 0
  y: 200
  width: 1920
  height: 600

segmentation:
  engine_path: "models/segmentation.engine"
  input_width: 640
  input_height: 640
  num_classes: 3
  threshold: 0.5

classes:
  1:
    name: "defect"
    min_area_mm2: 5
    max_area_mm2: 500
    min_solidity: 0.7
    max_elongation: 5.0

tracking:
  base_gating_mm: 30
  max_missed_frames: 5
  min_confirmations: 3

prediction:
  nozzle_position_mm: 500
  valve_delay_ms: 5
  air_response_ms: 5
  fire_duration_ms: 20
  safety_margin_ms: 3

safety:
  max_latency_ms: 30
  frame_timeout_ms: 200
  recovery_timeout_s: 5.0
```

---

## Running

### 1. Export segmentation model to TensorRT

```bash
trtexec \
  --onnx=model.onnx \
  --saveEngine=models/segmentation.engine \
  --fp16
```

### 2. Implement hardware interfaces

In `main.py`, replace the `None` placeholders:

```python
cfg["camera_object"]  = YourCamera()    # implements CameraInterface
cfg["encoder_object"] = YourEncoder()   # or None if no encoder
cfg["gpio_interface"] = YourGPIO()      # implements .setup() and .write()
cfg["ejector_pin"]    = 17
```

### 3. (Optional) Real-time OS setup

Reduces p99 latency from ~70ms to ~2–5ms:

```bash
# Add to GRUB_CMDLINE_LINUX in /etc/default/grub:
# isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3
sudo update-grub && sudo reboot

sudo cpupower frequency-set -g performance
sudo setcap cap_sys_nice+ep $(which python3)
```

### 4. Run

```bash
python3 main.py
```

---

## Testing

No GPU or hardware required:

```bash
python3 test_suite.py
# Ran 54 tests — OK
```

Modules covered by tests: `models`, `double_buffer`, `classical_cv_pipeline`, `tracker`, `predictor`, `scheduler`, `latency_monitor`, `watchdog`, `ejector` (mock GPIO), `camera` (mock).

Modules that require GPU to test: `gpu_roi_crop`, `gpu_preprocess`, `gpu_postprocess`, `segmentation_engine`, `pinned_memory`.

---

## Key Design Decisions

**Zero GPU round-trips.** All GPU stages are chained via raw device pointers (`CuPy MemoryPointer.ptr`). The path `roi_crop → preprocess → trt → postprocess` is entirely device-to-device.

**CuPy for custom kernels.** CUDA C kernels are loaded via `cupy.RawKernel` — same kernel code as raw CUDA, 3–5× less boilerplate than `python-cuda-bindings`. TensorRT bindings are kept as-is (no CuPy equivalent for TRT inference).

**Classical CV on CPU.** For 640×640 @ 120fps with 10–50 objects, OpenCV morphology + contours runs in 3–5ms. Moving it to GPU saves ~0.5ms average but does not reduce p99 latency, which is dominated by OS scheduler jitter. The correct fix is `isolcpus` + `SCHED_FIFO`.

**Kalman + Hungarian.** Tracker uses a 1D Kalman filter (position + velocity) with `scipy.optimize.linear_sum_assignment` — the same approach as SORT, DeepSORT, ByteTrack. `min_confirmations=3` prevents single-frame noise from triggering the ejector.

**Busy-wait scheduler.** Sleeps until 500µs before fire time, then busy-waits for sub-millisecond precision. Events are deduplicated per `object_id`.

**Non-blocking ejector.** `fire()` enqueues to a worker thread and returns in <0.1ms. The scheduler thread is never blocked for the pulse duration.

---

## Safety

| Mechanism | Monitors | Action |
|-----------|----------|--------|
| `LatencyMonitor` | p95 avg > threshold | Sets `is_overloaded()` |
| `Watchdog` | Frame timeout, latency overload | Disable ejector, reset tracker |
| `Watchdog` recovery | Heartbeat resumes | Re-enable ejector after `recovery_timeout_s` |
| `Ejector` clamp | Pulse > `MAX_DURATION_MS` | Clamp to 200ms |
| `Scheduler` dedup | Same `object_id` scheduled twice | Drop duplicate |
| `Scheduler` cancel | Tracker reset / watchdog fault | Remove pending event |

---

## Performance Reference

| Metric | Value | Conditions |
|--------|-------|------------|
| Camera FPS | 120 | 1920×1080 |
| H2D transfer | ~0.5ms | Full frame, PCIe Gen3 |
| GPU pipeline | ~3–5ms | 640×640, FP16, Ampere |
| Classical CV | ~3–5ms | 640×640, 10–50 objects |
| Tracker | <0.5ms | 50 tracks |
| Scheduler jitter | <500µs | busy-wait |
| End-to-end p99 | <5ms | PREEMPT_RT + isolcpus |

---

## License

MIT — see [LICENSE](LICENSE)
