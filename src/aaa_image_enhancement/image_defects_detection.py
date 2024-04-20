import cv2
import numpy as np
from skimage.exposure import is_low_contrast as ski_is_low_contrast
from skimage.restoration import estimate_sigma
from typing import NamedTuple
from PIL import Image, ImageChops


# можно сначала рассчитывать фичи изображения, а детекция уже происходит на основе фичей

# добавить фильтр на определение зернистости?

# по задумке я хотел бы сначал рассчитывать фичи (real),
# потом на их основе определять дефекты (bool)


class ImageProperties:
    def __init__(self, image):
        self.image = image
        self.gray = self.to_gray(image)
        self.trimmed_image = self.trim(self.cv2_to_pil(image))
        self.trimmed_gray = self.to_gray(self.pil_to_cv2(self.trimmed_image))

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

        return sigma > threshold

    def is_blurry(self, threshold=100.0) -> bool:
        blur_map = cv2.Laplacian(self.trimmed_gray, cv2.CV_64F)
        score = np.var(blur_map)
        return score < threshold

    def is_low_light(self, threshold=80) -> bool:
        avg_intensity = np.mean(self.trimmed_gray)
        return avg_intensity < threshold

    def is_low_contrast(self, threshold=0.35) -> bool:
        return ski_is_low_contrast(self.trimmed_image, fraction_threshold=threshold)

    def is_poor_white_balance(self):  # review
        b, g, r = cv2.split(self.pil_to_cv2(self.trimmed_image))
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        return (
            abs(avg_b - avg_g) > 20
            or abs(avg_b - avg_r) > 20
            or abs(avg_g - avg_r) > 20
        )


class ImageDefects(NamedTuple):
    blur: bool
    low_light: bool
    low_contrast: bool
    poor_white_balance: bool
    noisy: bool


def detect_image_defects(
    image_properties: ImageProperties,
    blur_threshold=100,
    low_light_threshold=80,
    contrast_threshold=0.35,
    noise_threshold=2.0,
) -> ImageDefects:
    defects = ImageDefects(
        blur=image_properties.is_blurry(blur_threshold),
        low_light=image_properties.is_low_light(low_light_threshold),
        low_contrast=image_properties.is_low_contrast(contrast_threshold),
        poor_white_balance=image_properties.is_poor_white_balance(),
        noisy=image_properties.is_noisy(noise_threshold),
    )
    return defects
