# cv/classical_cv_pipeline.py

import cv2
import numpy as np

from core.models import Detection
from core.interfaces import ClassicalCVInterface


class ClassicalCVPipeline(ClassicalCVInterface):
    """
    Classical computer vision pipeline.

    Responsibilities:
        - mask cleanup (morphology)
        - connected components / contour extraction
        - geometry analysis
        - shape metrics
        - filtering

    Constants precomputed in __init__:
        - mm_per_pixel_sq  (mm_per_pixel²) — avoids per-contour pow()
        - morphology kernels
    """

    def __init__(self, mm_per_pixel: float, class_configs: dict):

        self._mm_per_pixel    = mm_per_pixel
        self._mm_per_pixel_sq = mm_per_pixel * mm_per_pixel   # precomputed
        self._class_configs   = class_configs

        self._kernel_open  = np.ones((3, 3), np.uint8)
        self._kernel_close = np.ones((5, 5), np.uint8)

    # ============================================================
    # PROCESS
    # ============================================================

    def process(self, class_map: np.ndarray, timestamp_ns: int):

        detections = []

        mpp    = self._mm_per_pixel
        mpp_sq = self._mm_per_pixel_sq

        for class_id, cfg in self._class_configs.items():

            mask = (class_map == class_id).astype(np.uint8) * 255

            if cv2.countNonZero(mask) == 0:
                continue

            # ---- Morphology ----
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)

            # ---- Contours ----
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            min_area_mm2 = cfg["min_area_mm2"]
            max_area_mm2 = cfg["max_area_mm2"]
            min_solidity  = cfg["min_solidity"]
            max_elongation = cfg["max_elongation"]
            roi_x_min_mm  = cfg["roi_x_min_mm"]
            roi_x_max_mm  = cfg["roi_x_max_mm"]

            for cnt in contours:

                area_px = cv2.contourArea(cnt)

                if area_px <= 0:
                    continue

                area_mm2 = area_px * mpp_sq   # precomputed constant

                if area_mm2 < min_area_mm2 or area_mm2 > max_area_mm2:
                    continue

                # ---- Geometry ----
                x, y, w, h = cv2.boundingRect(cnt)

                width_mm  = w * mpp
                height_mm = h * mpp

                aspect_ratio = width_mm / (height_mm + 1e-6)
                elongation   = max(width_mm, height_mm) / (min(width_mm, height_mm) + 1e-6)

                cx_mm = (x + w * 0.5) * mpp
                cy_mm = (y + h * 0.5) * mpp

                # ---- Early reject before expensive hull ----
                if elongation > max_elongation:
                    continue

                if not (roi_x_min_mm <= cx_mm <= roi_x_max_mm):
                    continue

                # ---- Perimeter & hull ----
                perimeter = cv2.arcLength(cnt, True)

                hull          = cv2.convexHull(cnt)
                hull_area     = cv2.contourArea(hull) + 1e-6
                solidity      = area_px / hull_area

                if solidity < min_solidity:
                    continue

                hull_perimeter = cv2.arcLength(hull, True) + 1e-6
                convexity      = hull_perimeter / (perimeter + 1e-6)

                circularity    = (4.0 * np.pi * area_px) / (perimeter ** 2 + 1e-6)
                compactness    = (perimeter ** 2) / (area_px + 1e-6)

                detections.append(Detection(
                    x_mm         = cx_mm,
                    y_mm         = cy_mm,
                    class_id     = class_id,
                    area_mm2     = area_mm2,
                    width_mm     = width_mm,
                    height_mm    = height_mm,
                    aspect_ratio = aspect_ratio,
                    elongation   = elongation,
                    solidity     = solidity,
                    convexity    = convexity,
                    circularity  = circularity,
                    compactness  = compactness,
                    confidence   = 1.0,
                    timestamp_ns = timestamp_ns
                ))

        return detections
