"""Classes for applying enhancements based on found defects."""

from typing import Protocol

import numpy as np

from aaa_image_enhancement.enhancement_fns import (
    ENHANCEMENT_MAP,
    get_enhancement_function,
)
from aaa_image_enhancement.image_defects import DefectNames, ImageDefects


class ImageEnhancer:
    """Image enhancer to apply certain enhancement."""

    def __init__(self, img: np.ndarray) -> None:
        self.img = img

    def fix_defect(self, defect: DefectNames, **kwargs) -> np.ndarray:
        enhancement_fn = get_enhancement_function(defect)
        if enhancement_fn:
            enhanced_img = enhancement_fn(self.img, **kwargs)
            return enhanced_img
        else:
            raise ValueError(f"No enhancement function defined for {defect}")


class EnhanceStrategy(Protocol):
    """Agent to apply image enhancements using some rule."""

    def __init__(self, img: np.ndarray, defects: ImageDefects) -> None: ...
    def enhance_image(self) -> np.ndarray: ...


class EnhanceStrategyFirst:
    """Simple strategy to apply top priority enhancement."""

    def __init__(self, img: np.ndarray, defects: ImageDefects) -> None:
        """Pass image and ImageDefects."""
        self.defects = defects
        self.img = img
        self.priority_defects = list(ENHANCEMENT_MAP.keys())

    def enhance_image(self) -> np.ndarray:
        """Get enhanced image if there is a defect to fix. Or get the same image."""
        for defect in self.priority_defects:
            if self.defects.__dict__[defect.value]:
                enhanced_img = ImageEnhancer(self.img).fix_defect(defect)
                return np.array(enhanced_img)
        return self.img


class EnhanceStrategyMax:
    """Simple strategy to apply all enhancements."""

    def __init__(self, img: np.ndarray, defects: ImageDefects) -> None:
        """Pass image and ImageDefects."""
        self.defects = defects
        self.img = img
        self.priority_defects = list(ENHANCEMENT_MAP.keys())

    def enhance_image(self) -> np.ndarray:
        """Get enhanced image if there is a defect to fix. Or get the same image."""
        for defect in self.priority_defects:
            if self.defects.__dict__[defect.value]:
                self.img = ImageEnhancer(self.img).fix_defect(defect)
        return self.img
