from dataclasses import fields
from typing import Protocol
import numpy as np
import cv2
from cv2.typing import MatLike
import image_dehazer
from aaa_image_enhancement.exposure_enhancement import enhance_image_exposure
from aaa_image_enhancement.image_defects_detection import ImageDefects


class ImageEnhancer(Protocol):
    def __init__(self, img: np.ndarray, defects: ImageDefects) -> None: ...
    def enhance_image(self) -> np.ndarray: ...


class ImageEnhancerOpenCV:
    def __init__(self, img: np.ndarray, defects: ImageDefects) -> None:
        self.img = img
        self.defects = defects
        self.map_defect_fn = {
            "blur": self.deblur_image,
            "noisy": self.dehaze_image,
            "poor_white_balance": self.enhance_wb_image,
            "low_light": self.enhance_low_light,
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

    def enhance_image(self) -> np.ndarray:
        for defect in fields(self.defects):
            if getattr(self.defects, defect.name):
                enhancement_fn = self.map_defect_fn[defect.name]
                enhanced_image = enhancement_fn(self.img)
                return np.array(enhanced_image)
        print("no enhancement required")
        return self.img
