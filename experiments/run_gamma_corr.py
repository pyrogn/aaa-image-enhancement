import numpy as np
from PIL import Image

from src.aaa_image_enhancement.enhancement_fns import gamma_correction
from src.aaa_image_enhancement.image_utils import ImageConversions


def inference_gamma_enh(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    image_conv = ImageConversions(image)
    original_image = image_conv.to_cv2()
    return gamma_correction(original_image)
