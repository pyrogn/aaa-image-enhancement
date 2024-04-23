from dataclasses import dataclass
from enum import Enum
from typing import Callable

from aaa_image_enhancement.image_utils import ImageConversions


# описание, примеры и кандидаты на добавление находятся в гугл доке
@dataclass
class ImageDefects:
    """Image features for defect detection."""

    blur: bool = False
    low_light: bool = False
    low_contrast: bool = False
    poor_white_balance: bool = False
    noisy: bool = False
    hazy: bool = False
    jpeg_artifacts: bool = False
    glaring: bool = False
    rotation: bool = False


# add test to match enum value to key in dataclass?
class DefectNames(Enum):
    """Defect Enums to use in assignment and indexing.

    Value is a name of an ImageDefects dataclass attribute.
    """

    BLUR = "blur"
    LOW_LIGHT = "low_light"
    LOW_CONTRAST = "low_contrast"
    POOR_WHITE_BALANCE = "poor_white_balance"
    NOISY = "noisy"
    HAZY = "hazy"
    JPEG_ARTIFACTS = "jpeg_artifacts"
    GLARING = "glaring"
    ROTATION = "rotation"


# Почитать про протокол
# https://idego-group.com/blog/2023/02/21/we-need-to-talk-about-protocols-in-python/
# class DefectDetector(Protocol):
#     def __call__(self, image: ImageConversions, **kwargs) -> bool: ...


class DefectsDetector:
    def __init__(self, detectors: list[Callable]) -> None:
        """_summary_

        Args:
            detectors (list[Callable]): detectors should be sorted from least to most
                important, because later ones can override results
        """
        self.detectors = detectors

    def find_defects(self, image: ImageConversions, **kwargs) -> ImageDefects:
        defects = ImageDefects()
        for detector in self.detectors:
            # нельзя передать параметры, а надо ли?
            # если нужна функция, сам импортируешь и меняешь параметр
            detected_result = {k.value: v for k, v in detector(image).items()}
            defects.__dict__.update(detected_result)
        return defects
