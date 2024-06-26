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
def test_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
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


def test_image_defects_attributes():
    defects = ImageDefects()
    for defect in DefectNames:
        assert hasattr(defects, defect.value)


def test_image_defects_initialization():
    defects = ImageDefects(blur=True, low_light=False)
    assert defects.blur is True  # type: ignore
    assert defects.low_light is False  # type: ignore
    for defect in DefectNames:
        if defect.value not in ["blur", "low_light"]:
            assert getattr(defects, defect.value) is False
