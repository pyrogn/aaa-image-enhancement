import cv2
import numpy as np
from skimage.exposure import is_low_contrast as ski_is_low_contrast
from skimage.restoration import estimate_sigma
from typing import NamedTuple
from PIL import Image, ImageChops


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


# можно сначала рассчитывать фичи изображения, а детекция уже происходит на основе фичей

# добавить фильтр на определение зернистости?


def cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# я думал, что это обнаруживает дымку, но как-то плоховато работает
def is_noisy(image, threshold=2.0):  # review
    trimmed_image = trim(cv2_to_pil(image))
    trimmed_gray = to_gray(pil_to_cv2(trimmed_image))
    sigma = estimate_sigma(trimmed_gray, average_sigmas=True)
    return sigma > threshold


def is_blurry(image: np.array, threshold: int = 100):
    if image.ndim == 3:
        image = to_gray(image)
    trimmed_image = trim(cv2_to_pil(image))
    trimmed_gray = to_gray(pil_to_cv2(trimmed_image))
    blur_map = cv2.Laplacian(trimmed_gray, cv2.CV_64F)
    score = np.var(blur_map)
    # can return also blur_map, score
    return bool(score < threshold)


def is_low_light(image, threshold=80):
    trimmed_image = trim(cv2_to_pil(image))
    gray = to_gray(pil_to_cv2(trimmed_image))
    avg_intensity = np.mean(gray)
    return avg_intensity < threshold


def is_low_contrast(image, threshold=0.35):
    trimmed_image = trim(cv2_to_pil(image))
    return ski_is_low_contrast(trimmed_image, fraction_threshold=threshold)


def is_poor_white_balance(image):  # review
    trimmed_image = trim(cv2_to_pil(image))
    b, g, r = cv2.split(pil_to_cv2(trimmed_image))
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    return abs(avg_b - avg_g) > 20 or abs(avg_b - avg_r) > 20 or abs(avg_g - avg_r) > 20


class ImageDefects(NamedTuple):
    blur: bool
    low_light: bool
    low_contrast: bool
    poor_white_balance: bool
    noisy: bool


def detect_image_defects(
    image: np.array,
    blur_threshold=100,
    low_light_threshold=80,
    contrast_threshold=0.35,
    noise_threshold=2.0,
) -> ImageDefects:
    defects = ImageDefects(
        blur=is_blurry(image, blur_threshold),
        low_light=is_low_light(image, low_light_threshold),
        low_contrast=is_low_contrast(image, contrast_threshold),
        poor_white_balance=is_poor_white_balance(image),
        noisy=is_noisy(image, noise_threshold),
    )
    return defects
