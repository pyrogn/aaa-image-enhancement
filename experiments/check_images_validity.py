import glob

import cv2
from tqdm import tqdm


def is_valid_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        if len(img.shape) != 3 or img.shape[2] != 3:
            return False
        if img.shape[0] <= 50 or img.shape[1] <= 50:
            return False
        return True
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return False


if __name__ == "__main__":
    image_paths = glob.glob("./real_estate_images/*.jpg")
    valid_images = []
    invalid_images = []

    for image_path in tqdm(image_paths):
        if is_valid_image(image_path):
            valid_images.append(image_path)
        else:
            invalid_images.append(image_path)

    print(f"Total images: {len(image_paths)}")
    print(f"Valid images: {len(valid_images)}")
    print(f"Invalid images: {len(invalid_images)}")

    if invalid_images:
        print("Invalid image paths:")
        for image_path in invalid_images:
            print(image_path)
