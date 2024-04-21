from dataclasses import dataclass
from enum import Enum
import cv2
import numpy as np
from skimage.exposure import is_low_contrast as ski_is_low_contrast
from skimage.restoration import estimate_sigma
from typing import NamedTuple, Protocol
from PIL import Image, ImageChops


# можно сначала рассчитывать фичи изображения, а детекция уже происходит на основе фичей

# добавить фильтр на определение зернистости?

# по задумке я хотел бы сначал рассчитывать фичи (real),
# потом на их основе определять дефекты (bool)


# Добавить вертикальные фото
@dataclass
class ImageDefects:
    """Image features for defect detection."""

    blur: bool = False
    low_light: bool = False
    low_contrast: bool = False
    poor_white_balance: bool = False
    noisy: bool = False


class DefectNames(Enum):
    """Defect Enums to use in assignment and indexing.

    Value is a name of an ImageDefects dataclass attribute."""

    BLUR = "blur"
    LOW_LIGHT = "low_light"
    LOW_CONTRAST = "low_contrast"
    POOR_WHITE_BALANCE = "poor_white_balance"
    NOISY = "noisy"


# Почитать про протокол
# https://idego-group.com/blog/2023/02/21/we-need-to-talk-about-protocols-in-python/
class DefectsDetector(Protocol):
    def __init__(self, img: np.ndarray, **kwargs) -> None:
        "RGB image"

    def find_defects(self) -> ImageDefects: ...


class DefectsDetectorOpenCV:
    def __init__(self, img: np.ndarray, **kwargs) -> None:
        self.img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.gray = self.to_gray(img)
        self.trimmed_image = self.trim(self.cv2_to_pil(img))
        self.trimmed_gray = self.to_gray(self.pil_to_cv2(self.trimmed_image))
        self.params = kwargs

    @staticmethod
    def to_gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def trim(im):
        # https://stackoverflow.com/a/10616717
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)

    @staticmethod
    def cv2_to_pil(image):
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def pil_to_cv2(image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # я думал, что это обнаруживает дымку, но как-то плоховато работает
    def is_noisy(self, threshold=2.0) -> bool:  # review
        sigma = estimate_sigma(self.trimmed_gray, average_sigmas=True)

        return sigma > threshold  # type: ignore

    def is_blurry(self, threshold=100.0) -> bool:
        blur_map = cv2.Laplacian(self.trimmed_gray, cv2.CV_64F)
        score = np.var(blur_map)
        return score < threshold  # type: ignore

    def is_low_light(self, threshold=80) -> bool:
        avg_intensity = np.mean(self.trimmed_gray)
        return avg_intensity < threshold

    def is_low_contrast(self, threshold=0.35) -> bool:
        return ski_is_low_contrast(self.trimmed_image, fraction_threshold=threshold)

    def is_poor_white_balance(self):  # review
        b, g, r = cv2.split(self.pil_to_cv2(self.trimmed_image))
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)  # type: ignore
        return (
            abs(avg_b - avg_g) > 20
            or abs(avg_b - avg_r) > 20
            or abs(avg_g - avg_r) > 20
        )

    def find_defects(self) -> ImageDefects:
        defects = ImageDefects()
        # maybe add feature to selectively apply detectors?
        # is it too repetitive?
        defects.__dict__[DefectNames.BLUR.value] = self.is_blurry(
            self.params.get("blur_threshold", 100.0)
        )
        defects.__dict__[DefectNames.LOW_LIGHT.value] = self.is_low_light(
            self.params.get("low_light_threshold", 80)
        )
        defects.__dict__[DefectNames.LOW_CONTRAST.value] = self.is_low_contrast(
            self.params.get("low_contrast_threshold", 0.35)
        )
        defects.__dict__[DefectNames.POOR_WHITE_BALANCE.value] = (
            self.is_poor_white_balance()
        )
        defects.__dict__[DefectNames.NOISY.value] = self.is_noisy(
            self.params.get("noisy_threshold", 2.0)
        )
        return defects
