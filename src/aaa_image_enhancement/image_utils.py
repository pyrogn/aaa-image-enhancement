import cv2
import numpy as np
from PIL import Image, ImageChops


class ImageConversions:
    """Class helper for common format conversions.

    You can pass numpy RGB or PIL to __init__
    Or BGR to from_cv2() classmethod

    Convert image to grayscale using to_grayscale()

    Get result as
        RGB numpy array using to_numpy()
        BGR np array using to_cv2()
        PIL image using to_pil()
    """

    def __init__(self, img):
        """Init class

        Args:
            img: pass either numpy RGB or PIL Image.
        Attributes:
            img: RGB numpy
        """
        if isinstance(img, np.ndarray):
            self.img = img
        elif isinstance(img, Image.Image):
            self.img = self.pil_to_numpy(img)
        else:
            raise ValueError(
                "Unsupported image type. Expected numpy.ndarray, PIL.Image, or cv2.Mat."
            )

    @classmethod
    def from_cv2(cls, img):
        """Pass BGR numpy array."""
        return cls(cls.changeBR(img))

    @staticmethod
    def changeBR(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def to_numpy(self):
        """Returns RGB numpy array."""
        return self.img

    def to_pil(self):
        return Image.fromarray(self.img)

    def to_cv2(self) -> np.ndarray:
        if self.img.ndim == 2:
            return self.img
        return self.changeBR(self.img)

    @staticmethod
    def pil_to_numpy(pil_img):
        return np.array(pil_img)

    def to_grayscale(self):
        if len(self.img.shape) == 2:  # already grayscale
            return self.img
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

    # should it belong here or elsewhere?
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
