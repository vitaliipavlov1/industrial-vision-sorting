# inference/segmentation_engine.py

import logging

import numpy as np
import cupy as cp
import tensorrt as trt

from inference.gpu_preprocess import GPUPreprocessor
from core.interfaces import SegmentationEngineInterface


logger = logging.getLogger(__name__)


class SegmentationEngine(SegmentationEngineInterface):
    """
    TensorRT segmentation backend.

    Zero-copy GPU pipeline:
        infer(device_input_ptr, in_w, in_h)
            → GPUPreprocessor  (device→device: resize + norm + CHW)
            → TensorRT execute_async_v2
            → returns device pointer to probability maps

    Stream:
        One CuPy NonBlocking CUDA stream per engine.
        All operations run on this stream.
        Call synchronize() before reading results on CPU.

    CuPy replaces python-cuda-bindings for:
        - CUDA stream creation/sync/destroy
        - Device buffer allocation/free
        - cudaMemcpy for D2H copies
    TensorRT Python bindings remain (no CuPy equivalent).
    """

    def __init__(
        self,
        engine_path:  str,
        input_width:  int,
        input_height: int,
        num_classes:  int
    ):
        self.engine_path  = engine_path
        self.input_width  = input_width
        self.input_height = input_height
        self.num_classes  = num_classes
        self._destroyed   = False

        self._load_engine()
        self._create_stream()
        self._allocate_buffers()
        self._create_preprocessor()

        logger.info(
            "SegmentationEngine loaded: %s  input=%dx%d  classes=%d",
            engine_path, input_width, input_height, num_classes
        )

    # ============================================================
    # ENGINE
    # ============================================================

    def _load_engine(self) -> None:

        self._trt_logger = trt.Logger(trt.Logger.ERROR)
        self._runtime    = trt.Runtime(self._trt_logger)

        with open(self.engine_path, "rb") as f:
            engine_data = f.read()

        self._engine  = self._runtime.deserialize_cuda_engine(engine_data)
        self._context = self._engine.create_execution_context()

        if self._engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {self.engine_path}"
            )

    # ============================================================
    # STREAM  (CuPy)
    # ============================================================

    def _create_stream(self) -> None:
        # CuPy stream — replaces cudaStreamCreateWithFlags + cudaStreamNonBlocking
        self.stream = cp.cuda.Stream(non_blocking=True)

    # ============================================================
    # BUFFER ALLOCATION  (CuPy)
    # ============================================================

    def _allocate_buffers(self) -> None:

        self._bindings       = []
        self._device_inputs  = []
        self._device_outputs = []
        self._cupy_bufs      = []   # keep CuPy references alive

        for i in range(self._engine.num_bindings):

            dtype  = trt.nptype(self._engine.get_binding_dtype(i))
            shape  = self._context.get_binding_shape(i)
            size   = trt.volume(shape)
            nbytes = size * np.dtype(dtype).itemsize

            # CuPy device allocation — replaces cudaMalloc
            buf = cp.cuda.alloc(nbytes)
            self._cupy_bufs.append(buf)

            ptr = buf.ptr
            self._bindings.append(ptr)

            if self._engine.binding_is_input(i):
                self._device_inputs.append(ptr)
            else:
                self._device_outputs.append(ptr)

        logger.debug(
            "TRT buffers: %d inputs + %d outputs",
            len(self._device_inputs), len(self._device_outputs)
        )

    # ============================================================
    # PREPROCESSOR
    # ============================================================

    def _create_preprocessor(self) -> None:

        self._preprocessor = GPUPreprocessor(
            out_width  = self.input_width,
            out_height = self.input_height,
            channels   = 3
        )

    # ============================================================
    # INFER
    # ============================================================

    def infer(
        self,
        device_input_ptr: int,
        in_width:         int,
        in_height:        int
    ) -> int:
        """
        Full preprocessing + TRT inference, entirely on GPU.

        Args:
            device_input_ptr: raw device pointer to ROI image (HWC uint8)
            in_width:         ROI width
            in_height:        ROI height

        Returns:
            raw device pointer to probability maps (float32)

        All ops enqueued on self.stream. Call synchronize() before reading.
        """

        self._preprocessor.execute(
            device_input_ptr  = device_input_ptr,
            in_width          = in_width,
            in_height         = in_height,
            device_output_ptr = self._device_inputs[0],
            stream            = self.stream
        )

        self._context.execute_async_v2(
            bindings      = self._bindings,
            stream_handle = self.stream.ptr
        )

        return self._device_outputs[0]

    # ============================================================
    # SYNCHRONIZE
    # ============================================================

    def synchronize(self) -> None:
        """
        Blocks until all GPU work on this stream is complete.
        Must be called before CPU reads GPU output.
        """
        # CuPy — replaces cudaStreamSynchronize
        self.stream.synchronize()

    # ============================================================
    # DESTROY
    # ============================================================

    def destroy(self) -> None:

        if self._destroyed:
            return

        self._preprocessor.destroy()
        self._cupy_bufs.clear()       # releases all device buffers
        # stream released automatically by CuPy GC

        self._destroyed = True
        logger.info("SegmentationEngine destroyed")

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass
