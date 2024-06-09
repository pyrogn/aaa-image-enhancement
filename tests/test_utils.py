import numpy as np
import pytest
from aaa_image_enhancement.image_utils import ImageConversions


@pytest.fixture
def rgb_image():
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


def test_conversion_pil_to_numpy(rgb_image):
    pil_image = ImageConversions(rgb_image).to_pil()
    numpy_image = ImageConversions(pil_image).to_numpy()
    assert np.array_equal(rgb_image, numpy_image)


def test_conversion_cv2_to_numpy(rgb_image):
    cv2_image = ImageConversions.changeBR(rgb_image)
    image_conversion = ImageConversions.from_cv2(cv2_image)
    numpy_image = image_conversion.to_numpy()
    assert np.array_equal(rgb_image, numpy_image)


def test_grayscale_conversion(rgb_image):
    image_conversion = ImageConversions(rgb_image)
    grayscale_image = image_conversion.to_grayscale()
    assert grayscale_image.ndim == 2


def test_trim_image(rgb_image):
    image_conversion = ImageConversions(rgb_image)
    trimmed_image = image_conversion.trim()
    assert isinstance(trimmed_image, ImageConversions)
