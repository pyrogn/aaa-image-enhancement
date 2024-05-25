from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from src.aaa_image_enhancement.image_utils import ImageConversions


# описание, примеры и кандидаты на добавление находятся в гугл доке
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
    DARK_LIGHT = "dark_light"
    DARK_HISTOGRAM = "dark_histogram"
    DARK_THRESHOLD = "dark_threshold"
    DARK_LOCAL_CONTRAST = "dark_local_contrast"
    DARK_EDGES = "dark_edges"
    DARK_ADAPTIVE_THRESHOLD = "dark_adaptive_threshold"
    DARK_V_CHANNEL = "dark_v_channel"
    DARK_BLOCKS = "dark_blocks"


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
