from skimage.exposure import is_low_contrast as ski_is_low_contrast
from skimage.restoration import estimate_sigma
import cv2
import numpy as np

from aaa_image_enhancement.image_utils import ImageConversions


# TODO: разобраться с типизацией, что здесь не так?
def is_noisy(image: ImageConversions, threshold: float = 2.0) -> bool:  # review
    sigma = estimate_sigma(image.to_grayscale(), average_sigmas=True)
    return sigma > threshold  # type: ignore


def is_blurry(image: ImageConversions, threshold: float = 100.0) -> bool:
    blur_map = cv2.Laplacian(image.to_grayscale(), cv2.CV_64F)
    score = np.var(blur_map)  # type: ignore
    return score < threshold


def is_low_light(image: ImageConversions, threshold: int = 80) -> bool:
    avg_intensity = np.mean(image.to_grayscale())  # type: ignore
    return avg_intensity < threshold


def is_low_contrast(image: ImageConversions, threshold: float = 0.35) -> bool:
    return ski_is_low_contrast(image.to_numpy(), fraction_threshold=threshold)


def is_poor_white_balance(image: ImageConversions) -> bool:  # review
    b, g, r = cv2.split(image.to_cv2())
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)  # type: ignore
    return abs(avg_b - avg_g) > 20 or abs(avg_b - avg_r) > 20 or abs(avg_g - avg_r) > 20


# можно также собрать через декоратор, чем вручную
classical_detectors = [
    is_noisy,
    is_blurry,
    is_low_light,
    is_low_contrast,
    is_poor_white_balance,
]
