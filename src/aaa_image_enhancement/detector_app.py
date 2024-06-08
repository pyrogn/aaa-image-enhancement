import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile

from aaa_image_enhancement.detection_fns import is_low_light
from aaa_image_enhancement.image_defects import DefectsDetector
from aaa_image_enhancement.image_utils import ImageConversions

# we include only low light detection for now
detector = DefectsDetector([is_low_light])

app = FastAPI()


@app.post("/get_defects")
async def find_defects(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    defects_class = detector.find_defects(ImageConversions(img))
    defects_str = [defect for defect, value in defects_class.__dict__.items() if value]
    return defects_str
