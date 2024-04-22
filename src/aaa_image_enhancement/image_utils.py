import cv2
import numpy as np
from PIL import Image, ImageChops


class ImageConversions:
    """Class helper for common format conversions."""

    def __init__(self, img):
        if isinstance(img, np.ndarray):
            self.img = img
        elif isinstance(img, Image.Image):
            self.img = self.pil_to_numpy(img)
        elif isinstance(img, cv2.Mat):
            self.img = self.cv2_to_numpy(img)
        else:
            raise ValueError(
                "Unsupported image type. Expected numpy.ndarray, PIL.Image, or cv2.Mat."
            )

    def to_grayscale(self):
        if len(self.img.shape) == 2:  # already grayscale
            return self.img
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

    def trim(self, fuzz=0):
        """Trim image padding.

        Won't work if there are some text at the edges!!!
        https://stackoverflow.com/a/10616717
        """
        pil_img = self.to_pil()
        bg = Image.new(pil_img.mode, pil_img.size, pil_img.getpixel((0, 0)))
        diff = ImageChops.difference(pil_img, bg)
        if fuzz > 0:
            diff = ImageChops.add(diff, diff, 2.0, -fuzz)
        bbox = diff.getbbox()
        if bbox:
            self.img = self.pil_to_numpy(pil_img.crop(bbox))
        return self

    def to_numpy(self):
        if self.img.ndim == 2:
            return self.img
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def to_pil(self):
        return Image.fromarray(self.to_numpy())

    def to_cv2(self):
        return cv2.cvtColor(self.to_numpy(), cv2.COLOR_RGB2BGR)

    @staticmethod
    def pil_to_numpy(pil_img):
        return np.array(pil_img)

    @staticmethod
    def cv2_to_numpy(cv2_img):
        return cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
