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
    if params.is_blurry:  # handle somehow jpeg blocks, now it doesn't work
        # Apply Gaussian blur to smooth out artifacts
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        image = blurred
    # White Balance Correction
    wb = cv2.xphoto.createSimpleWB()
    white_balanced = wb.balanceWhite(image)

    # Contrast Enhancement using CLAHE
    lab = cv2.cvtColor(white_balanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    contrast_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Brightness Adjustment using Gamma Correction
    gamma = 1.2
    brightness_adjusted = cv2.pow(contrast_enhanced / 255.0, 1 / gamma) * 255.0
    brightness_adjusted = np.clip(brightness_adjusted, 0, 255).astype(np.uint8)

    # Sharpening using Unsharp Masking
    blurred = cv2.GaussianBlur(brightness_adjusted, (0, 0), 2.0)
    sharpened = cv2.addWeighted(brightness_adjusted, 1.5, blurred, -0.5, 0)

    # Noise Reduction using Bilateral Filtering
    noise_reduced = cv2.bilateralFilter(sharpened, 5, 75, 75)

    # Tone Mapping using Reinhard's Photographic Tone Reproduction
    tonemapped = cv2.createTonemapReinhard(
        gamma=1.0, intensity=0.0, light_adapt=1.0, color_adapt=0.0
    )
    tonemapped_image = tonemapped.process(noise_reduced.astype(np.float32))
    tonemapped_image = np.clip(tonemapped_image * 255, 0, 255).astype(np.uint8)

    # Detail Enhancement using Guided Filtering
    guided_filter = cv2.ximgproc.createGuidedFilter(tonemapped_image, radius=5, eps=0.1)
    enhanced_image = guided_filter.filter(tonemapped_image, tonemapped_image)

    return enhanced_image
