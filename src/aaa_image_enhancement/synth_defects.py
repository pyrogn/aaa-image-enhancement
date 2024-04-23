import random
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from aaa_image_enhancement.image_defects_detection import DefectNames
from aaa_image_enhancement.image_utils import ImageConversions


class ImageDistortions:
    """Add distortions to clean image and get decent synth data."""

    def __init__(self, img):
        self.img_conv = ImageConversions(img)
        self.img = self.img_conv.to_numpy()
        self.distortion_methods = {
            DefectNames.BLUR: self.blur,
            DefectNames.LOW_LIGHT: self.low_light,
            DefectNames.POOR_WHITE_BALANCE: self.poor_white_balance,
            DefectNames.HAZY: self.haziness,
            DefectNames.GLARING: self.brightness,
            DefectNames.LOW_CONTRAST: self.low_contrast,
            DefectNames.JPEG_ARTIFACTS: self.jpeg_artifacts,
            DefectNames.ROTATION: self.rotation,
        }
        self.applied_distortions = []

    def apply_distortions(self, distortion_types) -> tuple[np.ndarray, list[str]]:
        for distortion_type in distortion_types:
            if distortion_type in self.distortion_methods:
                self.img = self.distortion_methods[distortion_type]()
                self.applied_distortions.append(distortion_type)
            else:
                raise ValueError(f"Invalid distortion type: {distortion_type}")
        return self.img, self.applied_distortions

    def rotation(self):
        # если использовать, то надо научиться делать кроп,
        # чтобы не обучиться на черный цвет
        # можно зеркалить, но это тоже несет риски
        angle = random.uniform(20, 50)
        if random.random() < 0.5:
            angle = -angle

        im = self.img_conv.numpy_to_pil(self.img)
        im = im.rotate(angle)
        return self.img_conv.pil_to_numpy(im)

    def blur(self):
        if random.random() < 0.5:
            img = self._gaussian_blur()
        else:
            img = self._motion_blur()
        return img

    def _gaussian_blur(self):
        kernel_size = random.choice([7, 9, 11])
        return cv2.GaussianBlur(self.img, (kernel_size, kernel_size), 0)

    def _motion_blur(self):
        kernel_size = random.choice([7, 9, 11])
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel_motion_blur /= kernel_size
        return cv2.filter2D(self.img, -1, kernel_motion_blur)

    def low_light(self):
        brightness = random.uniform(0.1, 0.4)
        return cv2.convertScaleAbs(self.img, alpha=brightness, beta=0)

    def poor_white_balance(self):
        r_gain = random.uniform(0.6, 1.4)
        b_gain = random.uniform(0.6, 1.4)
        img_copy = self.img.copy()
        img_copy[:, :, 0] = cv2.multiply(img_copy[:, :, 0], r_gain)  # type: ignore
        img_copy[:, :, 2] = cv2.multiply(img_copy[:, :, 2], b_gain)  # type: ignore
        return img_copy

    def haziness(self):
        A = random.uniform(0.6, 0.9)  # Atmospheric light
        t = random.uniform(0.2, 0.7)  # Transmission map
        hazy_img = cv2.addWeighted(
            self.img.astype(np.float32),
            t,
            np.ones_like(self.img, dtype=np.float32) * 255 * A,
            1 - t,
            0,
        )
        return hazy_img.astype(np.uint8)

    def brightness(self):
        # можно ещё добавить методы, чтобы лучше имитировать различные световые дефекты
        brightness = random.uniform(1.5, 2.5)
        return cv2.convertScaleAbs(self.img, alpha=brightness, beta=0)

    def low_contrast(self):
        contrast = random.uniform(0.3, 0.7)
        pil_img = self.img_conv.to_pil()
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast)
        return self.img_conv.pil_to_numpy(pil_img)

    def jpeg_artifacts(self):
        quality = random.randint(5, 40)
        pil_img = self.img_conv.numpy_to_pil(self.img)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        pil_img = Image.open(buffer)
        return self.img_conv.pil_to_numpy(pil_img)
