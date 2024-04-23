import glob

import cv2
import numpy as np
import pytest
from aaa_image_enhancement.enhancement_fns import classical_enhancement_fns
from aaa_image_enhancement.image_utils import ImageConversions


@pytest.fixture
def image_path():
    return glob.glob("./real_estate_images/*.jpg")[0]


@pytest.fixture
def test_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    return ImageConversions(img).to_numpy()


@pytest.fixture
def image(image_path):
    img = cv2.imread(image_path)
    return ImageConversions(img)


@pytest.fixture
def numpy_image(image_path):
    return cv2.imread(image_path)


@pytest.mark.parametrize("enhancer", classical_enhancement_fns)
def test_enhancers(test_image, enhancer):
    enhanced_image = enhancer(test_image)
    assert isinstance(enhanced_image, np.ndarray)
    assert test_image.shape == enhanced_image.shape
