"""Add synthetic distortions to clean images to make a dataset."""

import random
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from aaa_image_enhancement.image_defects_detection import DefectNames
from aaa_image_enhancement.image_utils import ImageConversions


# можно смело добавлять больше разнообразия в дефекты,
# так как здесь пока чаще только 1 алгоритм применяется
# и также они должны быть приближены к реальным дефектам
class ImageDistortions:
    """Add distortions to clean image and get decent synth data."""

    def __init__(self, img):
        self.img_conv = ImageConversions(img)  # get RGB numpy
        self.img_np_rgb = self.img_conv.to_numpy()  # convert to cv2
        self.distortion_methods = {
            DefectNames.BLUR: self.blur,
            DefectNames.LOW_LIGHT: self.low_light,
            DefectNames.POOR_WHITE_BALANCE: self.poor_white_balance,
            DefectNames.HAZY: self.haziness,
            DefectNames.NOISY: self.noisy,
            DefectNames.GLARING: self.brightness,
            DefectNames.LOW_CONTRAST: self.low_contrast,
            DefectNames.JPEG_ARTIFACTS: self.jpeg_artifacts,
            DefectNames.ROTATION: self.rotation,
        }
        self.applied_distortions = []

    @property
    def img_np_bgr(self):
        """In case you need exactly cv2"""
        return ImageConversions.changeBR(self.img_np_rgb)

    def apply_distortions(
        self, distortion_types
    ) -> tuple[np.ndarray, list[DefectNames]]:
        for distortion_type in distortion_types:
            if distortion_type in self.distortion_methods:
                self.img_np_rgb = self.distortion_methods[distortion_type]()
                self.applied_distortions.append(distortion_type)
            else:
                raise ValueError(f"Invalid distortion type: {distortion_type}")
        return self.img_np_rgb, self.applied_distortions

    def noisy(self):
        """Add random Gaussian noise to the image."""
        mean = 0
        var = random.uniform(0.01, 0.02)
        sigma = var**0.5
        gaussian = np.random.normal(mean, sigma, self.img_np_rgb.shape)
        noisy_img = self.img_np_rgb + gaussian * 255  # type: ignore
        noisy_img = np.clip(noisy_img, 0, 255)
        return noisy_img.astype(np.uint8)

    def rotation(self):
        # если использовать, то надо научиться делать кроп,
        # чтобы не обучиться на черный цвет (так и происходит)
        # можно зеркалить, но это тоже несет риски
        angle = random.uniform(20, 50)
        if random.random() < 0.5:
            angle = -angle

        im = self.img_conv.to_pil()
        im = im.rotate(angle)
        return self.img_conv.pil_to_numpy(im)

    def blur(self):
        blur_type = random.choice(
            ["gaussian", "motion", "average", "median", "bilateral"]
        )
        # обратить внимание, что bilateral вряд ли можно исправить деблюром
        # возможно, ещё  другие. После деблюра получится аниме, а не фото.

        # blur_type = "bilateral"
        if blur_type == "gaussian":
            img = self._gaussian_blur()
        elif blur_type == "motion":
            img = self._motion_blur()
        elif blur_type == "average":
            img = self._average_blur()
        elif blur_type == "median":
            img = self._median_blur()
        elif blur_type == "bilateral":
            img = self._bilateral_blur()
        return img

    def _average_blur(self):
        kernel_size = random.choice([3, 5, 7, 9, 11])
        return cv2.blur(self.img_np_rgb, (kernel_size, kernel_size))

    def _median_blur(self):
        kernel_size = random.choice([3, 5, 7, 9, 11])
        return cv2.medianBlur(self.img_np_rgb, kernel_size)

    def _bilateral_blur(self):
        # самый странный фильтр. Он оставляет только линии и края,
        # остальной шум сглаживает.
        diameter = random.choice([5, 7, 9, 11])
        sigma_color = random.uniform(50, 100)
        sigma_space = random.uniform(50, 100)
        return cv2.bilateralFilter(self.img_np_rgb, diameter, sigma_color, sigma_space)

    def _gaussian_blur(self):
        kernel_size = random.choice([7, 9, 11])
        return cv2.GaussianBlur(self.img_np_rgb, (kernel_size, kernel_size), 0)

    def _motion_blur(self):
        kernel_size = random.choice([7, 9, 11])
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel_motion_blur /= kernel_size

        # Random angle for motion blur
        angle = random.randint(0, 360)
        rotation_matrix = cv2.getRotationMatrix2D(
            (kernel_size / 2, kernel_size / 2), angle, 1
        )
        kernel_motion_blur = cv2.warpAffine(
            kernel_motion_blur, rotation_matrix, (kernel_size, kernel_size)
        )

        return cv2.filter2D(self.img_np_rgb, -1, kernel_motion_blur)

    def low_light(self):
        # Gently reduce the brightness
        brightness = random.uniform(0.5, 0.9)
        dimmed_img = cv2.convertScaleAbs(self.img_np_rgb, alpha=brightness, beta=0)

        # # Slightly adjust the contrast
        # contrast = random.uniform(0.8, 1.0)
        # dimmed_img = cv2.convertScaleAbs(dimmed_img, alpha=contrast, beta=0)

        # Apply a subtle color shift to simulate the color temperature of low light
        # Typically, low light can cause a cooler look (bluish tint)
        b_gain = random.uniform(0.95, 1.05)  # Slight blue gain
        g_gain = random.uniform(0.95, 1.05)  # Slight green gain
        r_gain = random.uniform(0.90, 1.00)  # Slightly less red gain

        # Apply color gains
        dimmed_img[:, :, 0] = cv2.multiply(dimmed_img[:, :, 0], b_gain)  # type: ignore
        dimmed_img[:, :, 1] = cv2.multiply(dimmed_img[:, :, 1], g_gain)  # type: ignore
        dimmed_img[:, :, 2] = cv2.multiply(dimmed_img[:, :, 2], r_gain)  # type: ignore

        return dimmed_img

    def poor_white_balance(self):
        # Random gains for red, green, and blue channels
        r_gain = random.uniform(0.8, 1.2)
        g_gain = random.uniform(0.8, 1.2)
        b_gain = random.uniform(0.8, 1.2)
        img_copy = self.img_np_rgb.copy()

        # Apply gains to each channel
        img_copy[:, :, 0] = cv2.multiply(img_copy[:, :, 0], b_gain)  # type: ignore
        img_copy[:, :, 1] = cv2.multiply(img_copy[:, :, 1], g_gain)  # type: ignore
        img_copy[:, :, 2] = cv2.multiply(img_copy[:, :, 2], r_gain)  # type: ignore

        return img_copy

    def haziness(self):
        # очень похоже на низкую контрастность
        A = random.uniform(0.8, 0.95)  # Atmospheric light
        t = random.uniform(0.5, 0.9)  # Transmission map
        hazy_img = cv2.addWeighted(
            self.img_np_rgb.astype(np.float32),
            t,
            np.ones_like(self.img_np_rgb, dtype=np.float32) * 255 * A,
            1 - t,
            0,
        )
        return hazy_img.astype(np.uint8)

    def _apply_brightness(self):
        brightness = random.uniform(1.2, 1.5)
        return cv2.convertScaleAbs(self.img_np_rgb, alpha=brightness, beta=0)

    def _apply_overexposure(self):
        # looks like haze
        # Increase the brightness of the entire image to simulate overexposure
        overexposed_img = cv2.convertScaleAbs(
            self.img_np_rgb, alpha=1.0, beta=50
        )  # Increase beta to make the image brighter
        return overexposed_img

    def _apply_bloom(self):
        gray = cv2.cvtColor(self.img_np_rgb, cv2.COLOR_BGR2GRAY)
        _, bright_areas = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        bright_mask = cv2.cvtColor(bright_areas, cv2.COLOR_GRAY2BGR)
        increased_brightness = cv2.addWeighted(
            self.img_np_rgb, 1.5, bright_mask, 0.5, 0
        )
        return increased_brightness

    def brightness(self):
        chosen_fn = random.choice(
            [
                self._apply_brightness,
                self._apply_bloom,
                self._apply_overexposure,
            ]
        )
        # chosen_fn = self.apply_overexposure
        # print(chosen_fn)
        return chosen_fn()

    def low_contrast(self):
        # очень похоже на дымку
        contrast = random.uniform(0.5, 0.9)
        pil_img = self.img_conv.to_pil()
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast)
        return self.img_conv.pil_to_numpy(pil_img)

    def jpeg_artifacts(self):
        quality = random.randint(5, 50)
        pil_img = self.img_conv.to_pil()
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        pil_img = Image.open(buffer)
        return self.img_conv.pil_to_numpy(pil_img)
