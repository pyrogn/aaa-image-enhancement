"""File with functions for image enhancement for specific defects."""

import cv2
import image_dehazer
import numpy as np
from cv2.typing import MatLike

from aaa_image_enhancement.exposure_enhancement import enhance_image_exposure
from aaa_image_enhancement.image_defects_detection import DefectNames


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


def enhance_low_light_1(image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    enhanced_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return enhanced_image


def enhance_low_light_2(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)  # noqa: E741
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image


def enhance_low_light_3(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    enhanced_image = cv2.LUT(image, table)
    return enhanced_image


def enhance_low_light_4(
    image: np.ndarray, gamma: float = 1.5, alpha: float = 1.2, beta: int = -10
) -> np.ndarray:
    # Gamma correction
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    corrected_image = cv2.LUT(image, table)

    # contrast enhancement (not so visible)
    enhanced_image = cv2.convertScaleAbs(corrected_image, alpha=alpha, beta=beta)

    return enhanced_image


def enhance_low_light_5(
    image: np.ndarray, brightness: int = 45, contrast: int = 19
) -> np.ndarray:
    # adjusting brightness and contrast
    # works not so great
    alpha = 1 + contrast / 100.0
    beta = brightness
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


# used for basic testing
classical_enhancement_fns = [
    deblur_image,
    dehaze_image,
    enhance_wb_image,
    enhance_low_light,
]

# sorted, actually, from most to least important
ENHANCEMENT_MAP = {
    DefectNames.BLUR: deblur_image,
    DefectNames.LOW_LIGHT: enhance_low_light,
    DefectNames.POOR_WHITE_BALANCE: enhance_wb_image,
    DefectNames.NOISY: dehaze_image,
}


def get_enhancement_function(defect: DefectNames):
    return ENHANCEMENT_MAP.get(defect)
