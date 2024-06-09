import cv2
import httpx
import numpy as np
import pytest
import pytest_asyncio
from aaa_image_enhancement.app import app


@pytest_asyncio.fixture
async def http_client():
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_image_bytes():
    # Create a simple black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    return img_encoded.tobytes()


def mock_detect_problems_response():
    return {"problems": ["low_light"]}


def mock_enhance_image_response():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    return img_encoded.tobytes()


def mock_fix_defect_response():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode(".jpg", img)
    return img_encoded.tobytes()


@pytest.mark.asyncio
async def test_detect_problems(http_client, test_image_bytes, monkeypatch):
    async def mock_post(url, *args, **kwargs):
        class MockResponse:
            def json(self):
                return mock_detect_problems_response()

            @property
            def status_code(self):
                return 200

        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    response = await http_client.post(
        "/detect_problems",
        files={"image": ("test_image.jpg", test_image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == mock_detect_problems_response()


@pytest.mark.asyncio
async def test_enhance_image(http_client, test_image_bytes, monkeypatch):
    async def mock_post(url, *args, **kwargs):
        class MockResponse:
            def json(self):
                if "get_defects" in url:
                    return mock_detect_problems_response()
                if "enhance_image" in url:
                    return mock_enhance_image_response()

            @property
            def content(self):
                return mock_enhance_image_response()

            @property
            def status_code(self):
                return 200

        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    response = await http_client.post(
        "/enhance_image",
        files={"image": ("test_image.jpg", test_image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.content == mock_enhance_image_response()


@pytest.mark.asyncio
async def test_fix_defect(http_client, test_image_bytes, monkeypatch):
    async def mock_post(url, *args, **kwargs):
        class MockResponse:
            @property
            def content(self):
                return mock_fix_defect_response()

            @property
            def status_code(self):
                return 200

        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    response = await http_client.post(
        "/fix_defect",
        files={"image": ("test_image.jpg", test_image_bytes, "image/jpeg")},
        data={"defect_to_fix": "low_light"},
    )

    assert response.status_code == 200
    assert response.content == mock_fix_defect_response()
