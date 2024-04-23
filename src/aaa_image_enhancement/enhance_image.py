from typing import Protocol

import numpy as np

from aaa_image_enhancement.enhancement_fns import get_enhancement_function
from aaa_image_enhancement.image_defects_detection import DefectNames, ImageDefects


class ImageEnhancer:
    """Image enhancer based on classical techniques"""

    def __init__(self, img: np.ndarray) -> None:
        self.img = img

    def fix_defect(self, img: np.ndarray, defect: DefectNames, **kwargs) -> np.ndarray:
        enhancement_fn = get_enhancement_function(defect)
        if enhancement_fn:
            enhanced_img = enhancement_fn(img, **kwargs)
            return enhanced_img
        else:
            raise ValueError(f"No enhancement function defined for {defect}")


# maybe we can have attribute applied_enhancements=list[str|DefectNames]
# to keep track of enhancements
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
        return self.img
