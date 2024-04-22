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
def image(image_path):
    img = cv2.imread(image_path)
    return ImageConversions(img)


@pytest.fixture
def numpy_image(image_path):
    return cv2.imread(image_path)


@pytest.mark.parametrize("enhancer", classical_enhancement_fns)
def test_enhancers(numpy_image, enhancer):
    enhanced_image = enhancer(numpy_image)
    assert isinstance(enhanced_image, np.ndarray)
    assert numpy_image.shape == enhanced_image.shape
