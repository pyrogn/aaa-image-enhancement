import glob

import cv2
import numpy as np
import pytest
from aaa_image_enhancement.enhancement_agents import EnhanceAgentFirst, ImageEnhancer
from aaa_image_enhancement.enhancement_fns import classical_enhancement_fns
from aaa_image_enhancement.image_defects import DefectNames, ImageDefects
from aaa_image_enhancement.image_utils import ImageConversions


@pytest.fixture
def image_path():
    return glob.glob("./data/real_estate_images/*.jpg")[0]


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


@pytest.fixture
def defects_all_false():
    return ImageDefects()


@pytest.fixture
def defects_with_blur():
    return ImageDefects(blur=True)


@pytest.fixture
def image_enhancer(test_image):
    return ImageEnhancer(test_image)


@pytest.mark.parametrize("enhancer", classical_enhancement_fns)
def test_enhancers(test_image, enhancer):
    enhanced_image = enhancer(test_image)
    assert isinstance(enhanced_image, np.ndarray)
    assert test_image.shape == enhanced_image.shape


def test_image_fix_defect(test_image):
    enhancer = ImageEnhancer(test_image)
    enhanced_image = enhancer.fix_defect(DefectNames.BLUR)
    assert isinstance(enhanced_image, np.ndarray)
    assert enhanced_image.shape == test_image.shape


def test_enhance_agent_first_no_defects(test_image, defects_all_false, image_enhancer):
    agent = EnhanceAgentFirst(test_image, defects_all_false)
    enhanced_image = agent.enhance_image()
    assert np.array_equal(enhanced_image, test_image)


def test_enhance_agent_first_with_defect(test_image, defects_with_blur, image_enhancer):
    agent = EnhanceAgentFirst(test_image, defects_with_blur)
    enhanced_image = agent.enhance_image()
    assert isinstance(enhanced_image, np.ndarray)
    assert enhanced_image.shape == test_image.shape
