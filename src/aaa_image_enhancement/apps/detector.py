"""
Detection service for image defects using FastAPI.

This service detects various defects in images and returns the detected defects.
"""

from aaa_image_enhancement.detection_fns import is_low_light
from aaa_image_enhancement.image_defects import DefectsDetector, ImageDefects
from aaa_image_enhancement.image_utils import ImageConversions, decode_image
from fastapi import FastAPI, File, UploadFile

# Мы включаем только детекцию низкой освещенности
# Другие детекторы (и соответствующее улучшение) сейчас работают плохо
detector = DefectsDetector([is_low_light])

app = FastAPI()


def find_defects_in_image(image_bytes: bytes) -> ImageDefects:
    """
    Find defects in the given image.

    Args:
        image_bytes (bytes): The image data in bytes.

    Returns:
        ImageDefects: The detected defects in the image.
    """
    img = decode_image(image_bytes)
    return detector.find_defects(ImageConversions(img))


@app.post("/get_defects")
async def find_defects(image: UploadFile = File(...)):
    """
    API endpoint to find defects in an uploaded image.

    Args:
        image (UploadFile): The uploaded image file.

    Returns:
        list[str]: List of detected defects as strings.
    """
    contents = await image.read()
    defects = find_defects_in_image(contents)
    defects_str = [
        defect for defect, is_detected in defects.__dict__.items() if is_detected
    ]
    return defects_str
