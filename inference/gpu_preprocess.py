# inference/gpu_preprocess.py

import logging

import cupy as cp


logger = logging.getLogger(__name__)


CUDA_KERNEL = r"""
extern "C" __global__
void preprocess_kernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int in_width,
    int in_height,
    int out_width,
    int out_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height)
        return;

    float scale_x = (float)in_width  / out_width;
    float scale_y = (float)in_height / out_height;

    float src_xf = (x + 0.5f) * scale_x - 0.5f;
    float src_yf = (y + 0.5f) * scale_y - 0.5f;

    int x0 = (int)src_xf;
    int y0 = (int)src_yf;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = max(0, min(x0, in_width  - 1));
    x1 = max(0, min(x1, in_width  - 1));
    y0 = max(0, min(y0, in_height - 1));
    y1 = max(0, min(y1, in_height - 1));

    float wx = src_xf - floorf(src_xf);
    float wy = src_yf - floorf(src_yf);

    wx = fmaxf(0.0f, fminf(1.0f, wx));
    wy = fmaxf(0.0f, fminf(1.0f, wy));

    int total   = out_width * out_height;
    int dst_idx = y * out_width + x;

    for (int c = 0; c < 3; c++)
    {
        float p00 = input[(y0 * in_width + x0) * 3 + c];
        float p10 = input[(y0 * in_width + x1) * 3 + c];
        float p01 = input[(y1 * in_width + x0) * 3 + c];
        float p11 = input[(y1 * in_width + x1) * 3 + c];

        float val = (p00 * (1.0f - wx) + p10 * wx) * (1.0f - wy)
                  + (p01 * (1.0f - wx) + p11 * wx) *        wy;

        output[c * total + dst_idx] = val / 255.0f;
    }
}
"""


class GPUPreprocessor:
    """
    GPU image preprocessing for TensorRT.

    Accepts a raw device pointer — zero CPU round-trips.

    Operations:
        - bilinear resize
        - normalization (÷255, range [0,1])
        - HWC → planar CHW

    Uses CuPy RawKernel instead of python-cuda-bindings (nvrtc pipeline).
    All operations are device-to-device. No allocations per frame.
    """

    def __init__(self, out_width: int, out_height: int, channels: int = 3):

        self.out_width  = out_width
        self.out_height = out_height
        self.channels   = channels

        self._kernel    = cp.RawKernel(CUDA_KERNEL, "preprocess_kernel")
        self._destroyed = False

        logger.info(
            "GPUPreprocessor initialized: output=%dx%d (device→device)",
            out_width, out_height
        )

    # ============================================================
    # EXECUTE  (device → device, zero allocations)
    # ============================================================

    def execute(
        self,
        device_input_ptr:  int,
        in_width:          int,
        in_height:         int,
        device_output_ptr: int,
        stream:            cp.cuda.Stream
    ) -> None:
        """
        Bilinear resize + normalize + HWC→CHW entirely on GPU.
        No host memory touched. No allocations.
        """

        block = (16, 16, 1)
        grid  = (
            (self.out_width  + 15) // 16,
            (self.out_height + 15) // 16,
            1
        )

        self._kernel(
            grid, block,
            (
                device_input_ptr,
                device_output_ptr,
                in_width,
                in_height,
                self.out_width,
                self.out_height
            ),
            stream=stream
        )

    # ============================================================
    # DESTROY
    # ============================================================

    def destroy(self) -> None:
        if not self._destroyed:
            self._destroyed = True
            logger.debug("GPUPreprocessor destroyed")

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass
