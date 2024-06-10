"""
Basic asynchronous backend in FastAPI.

This module defines three routes:
- /detect_problems:
    Input: image
    Output: list of detected defects
- /enhance_image:
    Input: image
    Output: enhanced image or status indicating no enhancement needed
- /fix_defect:
    Input: image, defect_name (str)
    Output: image with the specific enhancement applied
"""

import logging

import httpx
from fastapi import FastAPI, File, Form, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, RootModel

logging.basicConfig(level=logging.INFO)
app = FastAPI()


class DetectedDefects(RootModel):
    """Model for detected defects in an image."""

    root: list[str]


class EnhancementRequest(BaseModel):
    """Model for an enhancement request."""

    defect_to_fix: str


@app.post("/detect_problems", response_model=DetectedDefects)
async def detect_problems_route(image: UploadFile = File(...)):
    """
    Detect problems in the uploaded image.

    Args:
        image (UploadFile): The uploaded image file.

    Returns:
        DetectedProblems: A list of detected defects.
    """
    contents = await image.read()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://detector:8000/get_defects", files={"image": contents}
        )
    logging.debug(f"Detect problems response: {response.json()}")
    return response.json()


@app.post("/enhance_image")
async def enhance_image_route(image: UploadFile = File(...)) -> Response:
    """
    Enhance the image if defects are detected;
    otherwise, indicate no enhancement needed.

    Args:
        image (UploadFile): The image file to be processed.

    Returns:
        Response:
            - 200 OK: Returns the enhanced image.
            - 204 No Content: Returns if no enhancement is needed.
            - 400 Bad Request: Returns if the input data is invalid.
    """
    contents = await image.read()
    async with httpx.AsyncClient() as client:
        detect_response = await client.post(
            "http://detector:8000/get_defects", files={"image": contents}
        )
        defects = detect_response.json()
        logging.debug(f"Defects detected: {defects}")
        if not defects:
            return PlainTextResponse(status_code=204)

        enhance_response = await client.post(
            "http://enhancer:8000/enhance_image",
            files={"image": contents},
            data={"defects": defects},
        )

    return Response(content=enhance_response.content, media_type="image/jpeg")


@app.post("/fix_defect")
async def fix_defect_route(
    image: UploadFile = File(...), defect_to_fix: str = Form(...)
):
    """
    Fix a specific defect in the image.

    Args:
        image (UploadFile): The uploaded image file.
        defect_to_fix (str): The defect to fix.

    Returns:
        Response: The enhanced image or an error message.
    """
    contents = await image.read()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://enhancer:8000/fix_defect",
            files={"image": contents},
            data={"defect_to_fix": defect_to_fix},
        )
    if response.status_code == 400:
        return JSONResponse(status_code=400, content=response.json())
    return Response(content=response.content, media_type="image/jpeg")
