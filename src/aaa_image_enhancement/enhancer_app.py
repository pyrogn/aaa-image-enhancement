import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from aaa_image_enhancement.enhance_image import EnhanceAgentFirst, ImageEnhancer
from aaa_image_enhancement.image_defects_detection import DefectNames, ImageDefects

app = FastAPI()


@app.post("/enhance_image")
async def enhance_image_route(
    image: UploadFile = File(...), defects: list[str] = Form(...)
):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    defects_obj = ImageDefects(**{defect: True for defect in defects})
    enhance_agent = EnhanceAgentFirst(img, defects_obj)
    enhanced_img = enhance_agent.enhance_image()
    _, encoded_img = cv2.imencode(".jpg", enhanced_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


@app.post("/fix_defect")
async def fix_defect_route(
    image: UploadFile = File(...), defect_to_fix: str = Form(...)
):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        defect_enum = DefectNames.from_value(defect_to_fix)
        image_enhancer = ImageEnhancer(img)
        enhanced_img = image_enhancer.fix_defect(defect_enum)
    except (KeyError, ValueError):
        return JSONResponse(status_code=400, content={"detail": "Invalid defect name"})
    # except ValueError as e:
    #     return JSONResponse(status_code=400, content={"detail": str(e)})

    _, encoded_img = cv2.imencode(".jpg", enhanced_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
