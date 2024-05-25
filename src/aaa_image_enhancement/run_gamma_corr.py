from src.aaa_image_enhancement.enhancement_fns import enhance_low_light_3
import cv2
import os
import time
from PIL import Image
from typing import List, Dict
import glob
from collections import Counter
from tqdm import tqdm
from src.aaa_image_enhancement.image_utils import ImageConversions
import numpy as np


def inference_gamma_enh(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    image_conv = ImageConversions(image)
    original_image = image_conv.to_cv2()
    return enhance_low_light_3(original_image)
