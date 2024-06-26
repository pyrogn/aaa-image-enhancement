"""Functions for image enhancement for specific defects.

ENHANCEMENT_MAP maps image defect (Enum) with a function that fixes it.
"""

import cv2
import image_dehazer
import numpy as np
from cv2.typing import MatLike

from aaa_image_enhancement.external.exposure_enhancement import enhance_image_exposure
from aaa_image_enhancement.image_defects import DefectNames


def deblur_image(image: np.ndarray, sharpen_strength: int = 9) -> MatLike:
    # https://stackoverflow.com/a/58243090
    sharpen_kernel = np.array([[-1, -1, -1], [-1, sharpen_strength, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)
    return sharpened_image


def dehaze_image(image: np.ndarray, C0: int = 50, C1: int = 500) -> MatLike:
    # https://github.com/Utkarsh-Deshmukh/Single-Image-Dehazing-Python/tree/master
    return image_dehazer.remove_haze(
        image, showHazeTransmissionMap=False, C0=C0, C1=C1
    )[0]


def enhance_wb_image(
    image: np.ndarray,
    p: float = 0.2,
    clip_limit: float = 1,
    tile_grid_size: tuple = (8, 8),
) -> MatLike:
    wb = cv2.xphoto.createSimpleWB()
    wb.setP(p)
    white_balanced_image = wb.balanceWhite(image)

    lab = cv2.cvtColor(white_balanced_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)  # noqa: E741
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l)
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_image


def enhance_low_light(
    image: np.ndarray, gamma: float = 0.6, lambda_: float = 0.15
) -> np.ndarray:
    # https://github.com/pvnieo/Low-light-Image-Enhancement/tree/master
    return enhance_image_exposure(image, gamma=gamma, lambda_=lambda_)


def gamma_correction(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    enhanced_image = cv2.LUT(image, table)
    return enhanced_image


# used for basic testing
classical_enhancement_fns = [
    deblur_image,
    dehaze_image,
    enhance_wb_image,
    enhance_low_light,
]

# Sorted from most to least important
# one function can fix multiple defects.
# If you are to use ML model, then you should create map
# DefectNames => address of service with model that supports REST API
# and modify consequent code to work with local functions and remote calls.
ENHANCEMENT_MAP = {
    DefectNames.BLUR: deblur_image,
    DefectNames.LOW_LIGHT: gamma_correction,
    DefectNames.POOR_WHITE_BALANCE: enhance_wb_image,
    DefectNames.NOISY: dehaze_image,
}


def get_enhancement_function(defect: DefectNames):
    return ENHANCEMENT_MAP.get(defect)
