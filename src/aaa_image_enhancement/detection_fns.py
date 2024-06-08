"""File with functions for defects detection."""

import cv2
import numpy as np
from skimage.exposure import is_low_contrast as ski_is_low_contrast
from skimage.restoration import estimate_sigma

from aaa_image_enhancement.image_defects import DefectNames
from aaa_image_enhancement.image_utils import ImageConversions


# TODO: разобраться с типизацией, что здесь не так?
def is_noisy(
    image: ImageConversions, threshold: float = 2.0
) -> dict[DefectNames, bool]:  # review
    sigma = estimate_sigma(image.to_grayscale(), average_sigmas=True)
    return {DefectNames.NOISY: sigma > threshold}  # type: ignore


def is_blurry(
    image: ImageConversions, threshold: float = 100.0
) -> dict[DefectNames, bool]:
    blur_map = cv2.Laplacian(image.to_grayscale(), cv2.CV_64F)
    score = np.var(blur_map)  # type: ignore
    return {DefectNames.BLUR: score < threshold}


def is_low_light(
    image: ImageConversions, threshold: int = 115
) -> dict[DefectNames, bool]:
    """Underexposure detection. Threshold is picked after some analysis."""
    avg_intensity = np.mean(image.to_grayscale())  # type: ignore
    return {DefectNames.LOW_LIGHT: avg_intensity < threshold}


def is_low_contrast(
    image: ImageConversions, threshold: float = 0.35
) -> dict[DefectNames, bool]:
    return {
        DefectNames.LOW_CONTRAST: ski_is_low_contrast(
            image.to_numpy(), fraction_threshold=threshold
        )
    }


def is_poor_white_balance(image: ImageConversions) -> dict[DefectNames, bool]:  # review
    b, g, r = cv2.split(image.to_cv2())
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)  # type: ignore
    return {
        DefectNames.POOR_WHITE_BALANCE: abs(avg_b - avg_g) > 20
        or abs(avg_b - avg_r) > 20
        or abs(avg_g - avg_r) > 20
    }


# one model will return dict with multiple DefectNames

# можно также собрать через декоратор, чем вручную
# used for basic testing
classical_detectors = [
    is_noisy,
    is_blurry,
    is_low_light,
    is_low_contrast,
    is_poor_white_balance,
]
