"""Basic asynchronous backend in FastAPI

Routes:
- /detect_problems:
    input: image
    output: dictionary with defects

- /enhance_image
    input: image
    output: image (autonomously enhanced)

- /fix_defect
    input: image, defect_name (str)
    output: image (with specific enhancement)
"""

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aaa_image_enhancement.defects_detection_fns import (
    classical_detectors,
    is_blurry,
    is_low_contrast,
    is_low_light,
    is_noisy,
    is_poor_white_balance,
)
from aaa_image_enhancement.enhance_image import EnhanceAgentFirst, ImageEnhancer
from aaa_image_enhancement.enhancement_fns import (
    deblur_image,
    dehaze_image,
    enhance_low_light,
    enhance_wb_image,
)
from aaa_image_enhancement.image_defects_detection import (
    DefectNames,
    DefectsDetector,
    ImageDefects,
)
from aaa_image_enhancement.image_utils import ImageConversions

app = FastAPI()

detectors = {
    DefectNames.BLUR: is_blurry,
    DefectNames.NOISY: is_noisy,
    DefectNames.LOW_LIGHT: is_low_light,
    DefectNames.LOW_CONTRAST: is_low_contrast,
    DefectNames.POOR_WHITE_BALANCE: is_poor_white_balance,
}
defects_detector = DefectsDetector(classical_detectors)

map_defect_fn = {
    DefectNames.BLUR: deblur_image,
    DefectNames.NOISY: dehaze_image,
    DefectNames.POOR_WHITE_BALANCE: enhance_wb_image,
    DefectNames.LOW_LIGHT: enhance_low_light,
}


class DetectedProblems(BaseModel):
    problems: list[str]


class EnhancementRequest(BaseModel):
    defect_to_fix: DefectNames


def detect_defects(image: np.ndarray) -> ImageDefects:
    return defects_detector.find_defects(ImageConversions(image))


def enhance_image(image: np.ndarray, defects: ImageDefects) -> np.ndarray:
    image_enhancer = ImageEnhancer(image, map_defect_fn)
    enhance_agent = EnhanceAgentFirst(image, image_enhancer, defects)
    return enhance_agent.enhance_image()


def fix_specific_defect(image: np.ndarray, defect_to_fix: DefectNames) -> np.ndarray:
    enhancement_fn = map_defect_fn[defect_to_fix]
    return enhancement_fn(image)


@app.post("/detect_problems", response_model=DetectedProblems)
async def detect_problems_route(image: UploadFile = File(...)):
    """Given image it sends back found problems."""
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    defects = detect_defects(img)
    problems = [defect for defect, value in defects.__dict__.items() if value]
    return {"problems": problems}


@app.post("/enhance_image")
async def enhance_image_route(image: UploadFile = File(...)):
    """Given image it enhances image without user interaction."""
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    defects = detect_defects(img)
    enhanced_img = enhance_image(img, defects)
    _, encoded_img = cv2.imencode(".jpg", enhanced_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


@app.post("/fix_defect")
async def fix_defect_route(
    image: UploadFile = File(...), defect_to_fix: str = Form(...)
):
    """Given image and defect name it fixes specific defect for a picture."""
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        defect_enum = DefectNames[defect_to_fix]
        enhanced_img = fix_specific_defect(img, defect_enum)
    except KeyError:
        return JSONResponse(status_code=400, content={"detail": "Invalid defect name"})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    _, encoded_img = cv2.imencode(".jpg", enhanced_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
