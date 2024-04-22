from typing import Callable, Protocol

import numpy as np

from aaa_image_enhancement.image_defects_detection import DefectNames, ImageDefects


class ImageEnhancer:
    """Image enhancer based on classical techniques"""

    def __init__(
        self, img: np.ndarray, map_defect_fn: dict[DefectNames, Callable]
    ) -> None:
        self.img = img
        self.map_defect_fn = map_defect_fn

    def fix_defect(self, img: np.ndarray, defect: DefectNames, **kwargs) -> np.ndarray:
        enhancement_fn = self.map_defect_fn[defect]
        enhanced_img = enhancement_fn(img, **kwargs)
        return enhanced_img


# map_defect_fn example
# {
#     DefectNames.BLUR: deblur_image,
#     DefectNames.NOISY: dehaze_image,
#     DefectNames.POOR_WHITE_BALANCE: enhance_wb_image,
#     DefectNames.LOW_LIGHT: enhance_low_light,
#     # add low_contrast
# }
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
