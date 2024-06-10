"""Basic asynchronous backend in FastAPI.

Routes:
- /detect_problems:
    input: image
    output: dictionary with defects

- /enhance_image
    input: image
    output: image (automatically enhanced)

- /fix_defect
    input: image, defect_name (str)
    output: image (with specific enhancement)
"""

import logging

import httpx
from fastapi import FastAPI, File, Form, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
app = FastAPI()


class DetectedProblems(BaseModel):
    problems: list[str]


class EnhancementRequest(BaseModel):
    defect_to_fix: str


@app.post("/detect_problems", response_model=DetectedProblems)
async def detect_problems_route(image: UploadFile = File(...)):
    contents = await image.read()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://detector:8000/get_defects", files={"image": contents}
        )
    logging.debug(f"Detect problems response: {response.json()}")
    return response.json()


@app.post("/enhance_image")
async def enhance_image_route(image: UploadFile = File(...)):
    """
    Enhances the image if defects are detected;
    otherwise, indicates no enhancement needed.

    Parameters:
    - image (UploadFile): The image file to be processed.

    Responses:
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
    Fixes a certain defect in an image.

    Responses:
    - 200 OK: Returns the enhanced image.
    - 400 Bad Request: Input data is invalid (image or defect).
    - 5xx: Error in the service.
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