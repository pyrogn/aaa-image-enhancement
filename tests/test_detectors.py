import glob

import cv2
import numpy as np
import pytest
from aaa_image_enhancement.detection_fns import classical_detectors
from aaa_image_enhancement.image_defects import (
    DefectNames,
    DefectsDetector,
    ImageDefects,
)
from aaa_image_enhancement.image_utils import ImageConversions


@pytest.fixture
def image_path():
    return glob.glob("./data/real_estate_images/*.jpg")[0]


@pytest.fixture
def test_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    return ImageConversions(img)


@pytest.fixture
def image(image_path):
    img = cv2.imread(image_path)
    return ImageConversions(img)


@pytest.mark.parametrize("detector", classical_detectors)
def test_detectors(test_image, detector):
    result = detector(test_image)
    for defect, is_detected in result.items():
        assert isinstance(defect, DefectNames)
        # should we allow np.bool_ or simple bool?
        assert isinstance(is_detected, bool | np.bool_)


def test_class_detector(test_image):
    detector = DefectsDetector(classical_detectors)
    result = detector.find_defects(test_image)
    assert isinstance(result, ImageDefects)
    for k, v in result.__dict__.items():
        assert isinstance(k, str)
        assert isinstance(v, bool | np.bool_)
