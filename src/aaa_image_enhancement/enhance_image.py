from dataclasses import fields
from typing import Protocol
import numpy as np
import cv2
from cv2.typing import MatLike
import image_dehazer
from aaa_image_enhancement.exposure_enhancement import enhance_image_exposure
from aaa_image_enhancement.image_defects_detection import DefectNames, ImageDefects


class ImageEnhancer(Protocol):
    def __init__(self, img: np.ndarray) -> None: ...
    def fix_defect(self, img: np.ndarray, defect: DefectNames) -> np.ndarray: ...


class ImageEnhancerOpenCV:
    def __init__(self, img: np.ndarray) -> None:
        self.img = img
        self.map_defect_fn = {
            DefectNames.BLUR: self.deblur_image,
            DefectNames.NOISY: self.dehaze_image,
            DefectNames.POOR_WHITE_BALANCE: self.enhance_wb_image,
            DefectNames.LOW_LIGHT: self.enhance_low_light,
            # add low_contrast
        }

    def deblur_image(self, image, sharpen_strength=9) -> MatLike:
        # https://stackoverflow.com/a/58243090
        sharpen_kernel = np.array(
            [[-1, -1, -1], [-1, sharpen_strength, -1], [-1, -1, -1]]
        )
        sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)
        return sharpened_image

    def dehaze_image(self, image, C0=50, C1=500) -> MatLike:
        # https://github.com/Utkarsh-Deshmukh/Single-Image-Dehazing-Python/tree/master
        return image_dehazer.remove_haze(
            image, showHazeTransmissionMap=False, C0=C0, C1=C1
        )[
            0
        ]  # should I index it?

    def enhance_wb_image(
        self, image, p=0.2, clip_limit=1, tile_grid_size=(8, 8)
    ) -> MatLike:
        wb = cv2.xphoto.createSimpleWB()
        wb.setP(p)
        white_balanced_image = wb.balanceWhite(image)

        lab = cv2.cvtColor(white_balanced_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_l = clahe.apply(l)
        enhanced_lab = cv2.merge((enhanced_l, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_image

    def enhance_low_light(self, image, gamma=0.6, lambda_=0.15) -> np.ndarray:
        # https://github.com/pvnieo/Low-light-Image-Enhancement/tree/master
        return enhance_image_exposure(image, gamma=gamma, lambda_=lambda_)

    def fix_defect(self, img: np.ndarray, defect: DefectNames) -> np.ndarray:
        enhancement_fn = self.map_defect_fn[defect]
        enhanced_img = enhancement_fn(img)
        return enhanced_img


class EnhanceAgent(Protocol):
    """Agent to apply enhancements using some rule. E.g. 1 or 2 enhancements."""

    def __init__(
        self, img: np.ndarray, image_enhancer: ImageEnhancer, defects: ImageDefects
    ) -> None: ...
    def enhance_image(self) -> np.ndarray: ...


class EnhanceAgentFirst:
    """Simple strategy to apply top priority enhancement."""

    def __init__(
        self, img: np.ndarray, image_enhancer: ImageEnhancer, defects: ImageDefects
    ) -> None:
        self.defects = defects
        self.img = img
        self.image_enhancer = image_enhancer
        self.priority_defects = [
            DefectNames.BLUR,
            DefectNames.LOW_LIGHT,
            DefectNames.POOR_WHITE_BALANCE,
            DefectNames.NOISY,
        ]

    def enhance_image(self) -> np.ndarray:
        for defect in self.priority_defects:
            if self.defects.__dict__[defect.value]:
                enhanced_img = self.image_enhancer.fix_defect(self.img, defect)
                return np.array(enhanced_img)
        print("no enhancement required")
        return self.img
