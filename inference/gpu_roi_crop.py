# inference/gpu_roi_crop.py

import logging

import cupy as cp


logger = logging.getLogger(__name__)


CUDA_KERNEL = r"""
extern "C" __global__
void roi_crop(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int in_width,
    int in_height,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= roi_w || y >= roi_h)
        return;

    int in_x = roi_x + x;
    int in_y = roi_y + y;

    in_x = max(0, min(in_x, in_width  - 1));
    in_y = max(0, min(in_y, in_height - 1));

    int in_idx  = (in_y * in_width  + in_x) * 3;
    int out_idx = (y    * roi_w     + x)    * 3;

    output[out_idx + 0] = input[in_idx + 0];
    output[out_idx + 1] = input[in_idx + 1];
    output[out_idx + 2] = input[in_idx + 2];
}
"""


class GPURoiCrop:
    """
    GPU ROI cropping module.

    Crops the conveyor region from the full camera frame
    before neural inference. Fully device-to-device — zero CPU involvement.

    Uses CuPy RawKernel instead of python-cuda-bindings (nvrtc + cuLaunchKernel).
    """

    def __init__(self, roi_x: int, roi_y: int, roi_w: int, roi_h: int):

        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_w = roi_w
        self.roi_h = roi_h

        self._destroyed = False

        # CuPy compiles and caches the kernel automatically
        self._kernel = cp.RawKernel(CUDA_KERNEL, "roi_crop")

        # Persistent output buffer (allocated once)
        self._device_output = cp.cuda.alloc(roi_w * roi_h * 3)

        logger.info(
            "GPURoiCrop initialized: roi=(%d,%d)  size=%dx%d",
            roi_x, roi_y, roi_w, roi_h
        )

    # ============================================================
    # EXECUTE
    # ============================================================

    def execute(
        self,
        device_input_ptr: int,
        stream:           cp.cuda.Stream,
        in_width:         int,
        in_height:        int
    ) -> int:
        """
        Crops ROI from full frame. Fully device-to-device.

        Args:
            device_input_ptr: raw device pointer (int) to full frame HWC uint8
            stream:           CuPy CUDA stream
            in_width:         full frame width
            in_height:        full frame height

        Returns:
            raw device pointer (int) to cropped ROI HWC uint8
        """

        block = (16, 16, 1)
        grid  = (
            (self.roi_w + 15) // 16,
            (self.roi_h + 15) // 16,
            1
        )

        self._kernel(
            grid, block,
            (
                device_input_ptr,
                self._device_output.ptr,
                in_width,
                in_height,
                self.roi_x,
                self.roi_y,
                self.roi_w,
                self.roi_h
            ),
            stream=stream
        )

        return self._device_output.ptr

    # ============================================================
    # DESTROY
    # ============================================================

    def destroy(self) -> None:

        if self._destroyed:
            return

        self._device_output = None   # releases CuPy device buffer
        self._destroyed = True
        logger.debug("GPURoiCrop destroyed")

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass
