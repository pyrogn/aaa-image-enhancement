import numpy as np
import cv2
import image_dehazer
from aaa_image_enhancement.exposure_enhancement import enhance_image_exposure
from aaa_image_enhancement.image_defects_detection import ImageDefects


def deblur_image(image, sharpen_strength=9):
    # https://stackoverflow.com/a/58243090
    sharpen_kernel = np.array([[-1, -1, -1], [-1, sharpen_strength, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)
    return sharpened_image


def dehaze_image(image, C0=50, C1=500):
    # https://github.com/Utkarsh-Deshmukh/Single-Image-Dehazing-Python/tree/master
    return image_dehazer.remove_haze(image, showHazeTransmissionMap=False, C0=C0, C1=C1)


def enhance_wb_image(image, p=0.2, clip_limit=1, tile_grid_size=(8, 8)):
    wb = cv2.xphoto.createSimpleWB()
    wb.setP(p)
    white_balanced_image = wb.balanceWhite(image)

    lab = cv2.cvtColor(white_balanced_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l)
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_image


def enhance_low_light(image, gamma=0.6, lambda_=0.15):
    # https://github.com/pvnieo/Low-light-Image-Enhancement/tree/master
    return enhance_image_exposure(image, gamma=gamma, lambda_=lambda_)


map_defect_fn = {
    ImageDefects.blur: deblur_image,
    ImageDefects.noisy: dehaze_image,
    ImageDefects.poor_white_balance: enhance_wb_image,
    ImageDefects.low_light: enhance_low_light,
}


def enhance_image(image, defects: ImageDefects, **kwargs):
    for defect in defects:
        if defect:
            enhancement_fn = map_defect_fn[defect]
            enhancement_kwargs = {
                k: v
                for k, v in kwargs.items()
                # crazy thing, rewrite this
                if k in enhancement_fn.__code__.co_varnames
            }
            enhanced_image = enhancement_fn(image, **enhancement_kwargs)
            return enhanced_image
    print("no enhancement required")
    return image
