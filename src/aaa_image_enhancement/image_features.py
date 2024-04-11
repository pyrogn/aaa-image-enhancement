import cv2
import numpy as np
from typing import NamedTuple


class ImageParams(NamedTuple):
    brightness: float
    contrast: float
    is_blurry: bool
    blur_score: float
    color_dist: np.ndarray
    sharpness: float
    dynamic_range: int
    edges: np.ndarray

    @staticmethod
    def find_params(image_path):
        image = cv2.imread(image_path)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = hsv[..., 2].mean()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()

        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = fm < 100
        blur_score = fm

        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        color_dist = cv2.normalize(hist, hist).flatten()

        sharpness = fm

        dynamic_range = np.max(gray) - np.min(gray)

        edges = cv2.Canny(gray, 100, 200)

        return ImageParams(
            brightness=brightness,
            contrast=contrast,
            is_blurry=is_blurry,
            blur_score=blur_score,
            color_dist=color_dist,
            sharpness=sharpness,
            dynamic_range=dynamic_range,
            edges=edges,
        )


def enhance_image(image, params):
    # Perform white balancing
    wb = cv2.xphoto.createSimpleWB()
    white_balanced = wb.balanceWhite(image)

    # Adjust contrast
    alpha = 1.2  # Contrast control (1.0-3.0)
    adjusted = cv2.convertScaleAbs(white_balanced, alpha=alpha)

    # Apply HDR-like enhancement
    hdr_enhanced = cv2.detailEnhance(adjusted, sigma_s=12, sigma_r=0.15)

    return hdr_enhanced
