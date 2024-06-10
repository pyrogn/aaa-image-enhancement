"""Classes for applying enhancements based on found defects."""

from typing import Protocol

import numpy as np

from aaa_image_enhancement.enhancement_fns import (
    ENHANCEMENT_MAP,
    get_enhancement_function,
)
from aaa_image_enhancement.image_defects import DefectNames, ImageDefects


class ImageEnhancer:
    """
    Image enhancer to apply certain enhancement.

    Attributes:
        img (np.ndarray): The image to be enhanced.
    """

    def __init__(self, img: np.ndarray) -> None:
        self.img = img

    def fix_defect(self, defect: DefectNames, **kwargs) -> np.ndarray:
        """
        Apply the enhancement function for a given defect.

        Args:
            defect (DefectNames): The defect to be fixed.

        Returns:
            np.ndarray: The enhanced image.
        """
        enhancement_fn = get_enhancement_function(defect)
        if enhancement_fn:
            enhanced_img = enhancement_fn(self.img, **kwargs)
            return enhanced_img
        else:
            raise ValueError(f"No enhancement function defined for {defect}")


class EnhanceStrategy(Protocol):
    """
    Protocol for image enhancement strategies.

    Defines a blueprint for enhancement strategies that can be applied to an image.
    """

    def __init__(self, img: np.ndarray, defects: ImageDefects) -> None: ...
    def enhance_image(self) -> np.ndarray: ...


class EnhanceStrategyFirst:
    """Simple strategy to apply the top priority enhancement."""

    def __init__(self, img: np.ndarray, defects: ImageDefects) -> None:
        """Initialize the strategy with an image and its detected defects."""
        self.defects = defects
        self.img = img
        self.priority_defects = list(ENHANCEMENT_MAP.keys())

    def enhance_image(self) -> np.ndarray:
        """
        Apply the top priority enhancement if a defect is found.

        Returns:
            np.ndarray: The enhanced image,
                or the original image if no enhancements were applied.
        """
        for defect in self.priority_defects:
            if self.defects.__dict__[defect.value]:
                enhanced_img = ImageEnhancer(self.img).fix_defect(defect)
                return np.array(enhanced_img)
        return self.img


class EnhanceStrategyMax:
    """Simple strategy to apply all enhancements."""

    def __init__(self, img: np.ndarray, defects: ImageDefects) -> None:
        """Initialize the strategy with an image and its detected defects."""
        self.defects = defects
        self.img = img
        self.priority_defects = list(ENHANCEMENT_MAP.keys())

    def enhance_image(self) -> np.ndarray:
        """
        Apply the top priority enhancement if a defect is found.

        Returns:
            np.ndarray: The enhanced image,
                or the original image if no enhancements were applied.
        """
        for defect in self.priority_defects:
            if self.defects.__dict__[defect.value]:
                self.img = ImageEnhancer(self.img).fix_defect(defect)
        return self.img
