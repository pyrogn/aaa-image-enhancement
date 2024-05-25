from src.aaa_image_enhancement.image_utils import ImageConversions
from EnhanceIMG.retinex.enhancer import *
import numpy as np


def inference_enhance_img(img_path: str, sigma_list: list = None) -> np.ndarray:
    img_bgr = cv2.imread(img_path)

    img_conv = ImageConversions.from_cv2(img_bgr)
    img_rgb = img_conv.to_numpy()

    img_attnmsr = AttnMSR(img_rgb, sigma_list)  # Assuming AttnMSR is a function you have

    return img_attnmsr
