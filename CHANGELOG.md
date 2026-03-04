# Changelog

## [1.0.0] — 2025

### Initial production release

#### Architecture
- Zero-copy GPU pipeline: ROI crop → preprocess → TensorRT → postprocess, fully device-to-device
- Single H2D and D2H DMA transfer per frame (pinned memory)
- CuPy `RawKernel` for all custom CUDA kernels (replaces python-cuda-bindings)
- TensorRT FP16 inference via Python bindings

#### Tracking
- 1D Kalman filter (position + velocity state)
- Hungarian assignment via `scipy.optimize.linear_sum_assignment`
- `min_confirmations` guard — prevents single-frame noise from triggering ejector
- O(1) track lookup via `Dict[int, _Track]`

#### Scheduling
- Priority queue (`heapq`) with per-object deduplication
- Sleep + busy-wait strategy for <500µs jitter
- `cancel_object()` API for watchdog / tracker reset integration

#### Safety
- `LatencyMonitor`: p95/p99 with pre-allocated numpy circular buffer (zero allocs per frame)
- `Watchdog`: frame timeout + latency overload detection, auto-recovery, fault history
- `Ejector`: non-blocking worker thread, pulse clamping, queue drain on disable

#### Performance
- Removed per-frame `cudaMalloc` in preprocessor and main pipeline
- Precomputed `mm_per_pixel²` in ClassicalCV (was computed per contour)
- Early filter rejection before convex hull in ClassicalCV
- Thread-safe `DoubleBuffer` with `threading.Lock`
- All imports at module level — no imports inside hot-path functions

#### Testing
- 54 unit tests, no GPU or hardware required
- Mocks for GPIO, camera, ejector, tracker
- Fixed two tracker bugs found during testing:
  - New track with `min_confirmations=1` was not confirmed on first frame
  - New tracks created in the same update were incorrectly getting `missed_frames` incremented
