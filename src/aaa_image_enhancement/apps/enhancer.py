from aaa_image_enhancement.enhancement_strategies import (
    EnhanceStrategyFirst,
    ImageEnhancer,
)
from aaa_image_enhancement.image_defects import DefectNames, ImageDefects
from aaa_image_enhancement.image_utils import decode_image, encode_image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response


def enhance_image(image_bytes: bytes, defects: list[str]) -> bytes:
    img = decode_image(image_bytes)
    defects_obj = ImageDefects(**{defect: True for defect in defects})
    enhance_agent = EnhanceStrategyFirst(img, defects_obj)
    enhanced_img = enhance_agent.enhance_image()
    return encode_image(enhanced_img)


def fix_defect(image_bytes: bytes, defect_to_fix: str) -> bytes:
    img = decode_image(image_bytes)
    defect_enum = DefectNames(defect_to_fix)
    image_enhancer = ImageEnhancer(img)
    enhanced_img = image_enhancer.fix_defect(defect_enum)
    return encode_image(enhanced_img)


app = FastAPI()


@app.post("/enhance_image")
async def enhance_image_route(
    image: UploadFile = File(...), defects: list[str] = Form(...)
):
    contents = await image.read()
    try:
        enhanced_image_bytes = enhance_image(contents, defects)
        return Response(content=enhanced_image_bytes, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})


@app.post("/fix_defect")
async def fix_defect_route(
    image: UploadFile = File(...), defect_to_fix: str = Form(...)
):
    contents = await image.read()
    try:
        enhanced_image_bytes = fix_defect(contents, defect_to_fix)
        return Response(content=enhanced_image_bytes, media_type="image/jpeg")
    except (KeyError, ValueError):
        return JSONResponse(status_code=400, content={"detail": "Invalid defect name"})
