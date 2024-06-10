import cv2
import httpx
import numpy as np
import pytest
import pytest_asyncio


@pytest_asyncio.fixture()
async def http_client():
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def test_image_bytes():
    # Create a simple black image to simulate a low-light image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    return img_encoded.tobytes()


@pytest.mark.asyncio
async def test_enhance_image(http_client, test_image_bytes):
    response = await http_client.post(
        "http://enhancer:8000/enhance_image",
        files={"image": ("test_image.jpg", test_image_bytes, "image/jpeg")},
        data={"defects": ["low_light"]},
    )
    assert response.status_code == 200
    assert response.content  # Check that the response contains an image


@pytest.mark.asyncio
async def test_fix_low_light_defect(http_client, test_image_bytes):
    response = await http_client.post(
        "http://enhancer:8000/fix_defect",
        files={"image": ("test_image.jpg", test_image_bytes, "image/jpeg")},
        data={"defect_to_fix": "low_light"},
    )
    assert response.status_code == 200
    assert response.content  # Check that the response contains an image
