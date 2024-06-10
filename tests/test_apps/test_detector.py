# test_detector_service.py
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
    # Create a simple black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    return img_encoded.tobytes()


@pytest.mark.asyncio
async def test_find_defects(http_client, test_image_bytes):
    response = await http_client.post(
        "http://detector:8000/get_defects",
        files={"image": ("test_image.jpg", test_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)
