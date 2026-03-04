# inference/gpu_postprocess.py

import logging

import numpy as np
import cupy as cp


logger = logging.getLogger(__name__)


CUDA_KERNEL = r"""
extern "C" __global__
void fused_postprocess(
    const float* __restrict__ prob_maps,
    unsigned char* __restrict__ class_map,
    int num_classes,
    int width,
    int height,
    float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx   = y * width + x;
    int total = width * height;

    int   best_class = 0;
    float best_score = 0.0f;

    for (int c = 0; c < num_classes; c++)
    {
        float score = prob_maps[c * total + idx];
        if (score > best_score)
        {
            best_score = score;
            best_class = c;
        }
    }

    if (best_score < threshold)
        best_class = 0;

    class_map[idx] = (unsigned char)best_class;
}
"""


class GPUPostProcessor:
    """
    GPU segmentation postprocessing.

    Operations:
        - argmax over num_classes probability maps
        - confidence threshold (below threshold → class 0 = background)
        - output class_map (uint8, one byte per pixel)

    Uses CuPy RawKernel. copy_to_host() uses CuPy async D2H into
    pre-allocated pinned numpy array.
    """

    def __init__(self, width: int, height: int, num_classes: int):

        self.width        = width
        self.height       = height
        self.num_classes  = num_classes
        self.total_pixels = width * height

        self._kernel         = cp.RawKernel(CUDA_KERNEL, "fused_postprocess")
        self._device_class_map = cp.cuda.alloc(self.total_pixels)  # uint8
        self._destroyed      = False

        logger.info(
            "GPUPostProcessor initialized: %dx%d  classes=%d",
            width, height, num_classes
        )

    # ============================================================
    # EXECUTE
    # ============================================================

    def execute(
        self,
        device_prob_ptr: int,
        stream:          cp.cuda.Stream,
        threshold:       float
    ) -> int:
        """
        Runs argmax + threshold on device probability maps.

        Returns:
            raw device pointer (int) to class_map (uint8)
        """

        block = (16, 16, 1)
        grid  = (
            (self.width  + 15) // 16,
            (self.height + 15) // 16,
            1
        )

        self._kernel(
            grid, block,
            (
                device_prob_ptr,
                self._device_class_map.ptr,
                self.num_classes,
                self.width,
                self.height,
                np.float32(threshold)
            ),
            stream=stream
        )

        return self._device_class_map.ptr

    # ============================================================
    # COPY TO HOST  (async D2H into pinned memory)
    # ============================================================

    def copy_to_host(self, host_array: np.ndarray, stream: cp.cuda.Stream) -> None:
        """
        Async D2H copy into pre-allocated pinned host_array.
        Caller must call stream.synchronize() before reading host_array.
        """

        cp.cuda.runtime.memcpyAsync(
            host_array.ctypes.data,
            self._device_class_map.ptr,
            self.total_pixels,
            cp.cuda.runtime.memcpyDeviceToHost,
            stream.ptr
        )

    # ============================================================
    # DESTROY
    # ============================================================

    def destroy(self) -> None:

        if self._destroyed:
            return

        self._device_class_map = None   # releases CuPy device buffer
        self._destroyed = True
        logger.debug("GPUPostProcessor destroyed")

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass
