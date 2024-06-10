import cv2
import httpx
import numpy as np
import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def http_client():
    async with httpx.AsyncClient(base_url="http://main_app:8000") as client:
        yield client


@pytest.fixture
def test_image_bytes():
    # Create a simple black image
    img_black = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded_black = cv2.imencode(".jpg", img_black)

    # Create a simple white image
    img_white = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, img_encoded_white = cv2.imencode(".jpg", img_white)

    return img_encoded_black.tobytes(), img_encoded_white.tobytes()


@pytest.mark.asyncio
async def test_detect_problems(http_client, test_image_bytes):
    img_black, img_white = test_image_bytes

    # Test with black image
    response = await http_client.post(
        "/detect_problems",
        files={"image": ("test_image_black.jpg", img_black, "image/jpeg")},
    )
    assert response.status_code == 200

    # Test with white image
    response = await http_client.post(
        "/detect_problems",
        files={"image": ("test_image_white.jpg", img_white, "image/jpeg")},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_enhance_image(http_client, test_image_bytes):
    img_black, img_white = test_image_bytes

    # Test with black image
    response = await http_client.post(
        "/enhance_image",
        files={"image": ("test_image_black.jpg", img_black, "image/jpeg")},
    )
    assert response.status_code == 200 or response.status_code == 204

    # Test with white image
    response = await http_client.post(
        "/enhance_image",
        files={"image": ("test_image_white.jpg", img_white, "image/jpeg")},
    )
    assert response.status_code == 200 or response.status_code == 204


@pytest.mark.asyncio
async def test_fix_defect(http_client, test_image_bytes):
    img_black, img_white = test_image_bytes

    # Test with black image
    response = await http_client.post(
        "/fix_defect",
        files={"image": ("test_image_black.jpg", img_black, "image/jpeg")},
        data={"defect_to_fix": "low_light"},
    )
    assert response.status_code == 200

    # Test with white image
    response = await http_client.post(
        "/fix_defect",
        files={"image": ("test_image_white.jpg", img_white, "image/jpeg")},
        data={"defect_to_fix": "low_light"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_fix_defect_invalid_name(http_client, test_image_bytes):
    img_black, img_white = test_image_bytes

    # Test with black image
    response = await http_client.post(
        "/fix_defect",
        files={"image": ("test_image_black.jpg", img_black, "image/jpeg")},
        data={"defect_to_fix": "invalid_defect"},
    )
    assert response.status_code == 400

    # Test with white image
    response = await http_client.post(
        "/fix_defect",
        files={"image": ("test_image_white.jpg", img_white, "image/jpeg")},
        data={"defect_to_fix": "invalid_defect"},
    )
    assert response.status_code == 400
