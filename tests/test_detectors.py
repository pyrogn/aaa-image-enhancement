import glob
import cv2
import numpy as np
import pytest
from aaa_image_enhancement.image_utils import ImageConversions
from aaa_image_enhancement.defects_detection_fns import classical_detectors


@pytest.fixture
def image_path():
    return glob.glob("./real_estate_images/*.jpg")[0]


@pytest.fixture
def image(image_path):
    img = cv2.imread(image_path)
    return ImageConversions(img)


@pytest.mark.parametrize("detector", classical_detectors)
def test_detectors(image, detector):
    # should we allow np.bool_ or simple bool?
    assert isinstance(detector(image), (bool, np.bool_))
