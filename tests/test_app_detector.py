"""Tests for detector app. WIP!!!"""

import httpx
import pytest


@pytest.fixture
async def http_client():
    async with httpx.AsyncClient() as client:
        yield client


@pytest.mark.skip()
@pytest.mark.asyncio
async def test_find_defects(http_client):
    with open("test_image.jpg", "rb") as f:
        response = await http_client.post(
            "http://detector:8001/get_defects", files={"image": f}
        )
    assert response.status_code == 200
    assert "problems" in response.json()
