"""Basic classes for passing information about defects detection.

DefectNames is an Enum class for selecting a defect in general.

ImageDefects is a dataclass for passing information
on particular image defects between functions.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from aaa_image_enhancement.image_utils import ImageConversions


class DefectNames(Enum):
    """Defect Enums to use in assignment and indexing.

    Value is a name of an ImageDefects dataclass attribute.

    Описание, примеры и кандидаты на добавление находятся в гугл доке
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


@dataclass
class ImageDefects:
    """Image features for defect detection.

    It is used for passing information from detector to enhancer.
    """

    def __init__(self, **kwargs: bool) -> None:
        """Setup all defects. Default is False for every type in Enum."""
        for defect in DefectNames:
            setattr(self, defect.value, kwargs.get(defect.value, False))

    def __repr__(self) -> str:
        return (
            f"ImageDefects({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"
        )

    def has_defects(self) -> bool:
        """Returns true if found at least one defect."""
        return any(self.__dict__.values())


class DefectsDetector:
    """
    Class to detect defects in images using provided detection functions.

    Attributes:
        detectors (list[Callable]): List of detection functions.
    """

    def __init__(self, detectors: list[Callable[[ImageConversions], dict]]) -> None:
        """
        Initialize DefectsDetector with a list of detection functions.

        Args:
            detectors (list[Callable[[ImageConversions], dict]]): Detectors should be
                sorted from least to most important,
                because later ones can override results.
        """
        self.detectors = detectors

    def find_defects(self, image: ImageConversions, **kwargs) -> ImageDefects:
        """
        Detect defects in the given image.

        Args:
            image (ImageConversions): The image to be analyzed for defects.
            **kwargs: Additional arguments for detection functions.

        Returns:
            ImageDefects: The detected defects in the image.
        """
        defects = ImageDefects()
        for detector in self.detectors:
            # нельзя передать параметры, а надо ли?
            # если нужна функция, сам импортируешь и меняешь параметр
            detected_result = {k.value: v for k, v in detector(image).items()}
            defects.__dict__.update(detected_result)
        return defects
