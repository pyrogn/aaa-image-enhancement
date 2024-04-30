import os
import shutil
from tqdm import tqdm
from ocrmac import ocrmac
from pathlib import Path
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def process_images_directories(data_path: str):
    """
    Удаляет подпапки data_path с изображениями, перемещая все фото в папку data_path.
    """

    for root, dirs, files in tqdm(os.walk(data_path)):
        for file in files:

            file_path = os.path.join(root, file)

            destination_path = os.path.join(data_path, file)

            shutil.move(file_path, destination_path)

            print(f'Файл {file_path} перемещен в {destination_path}')

    for root, dirs, files in tqdm(os.walk(data_path, topdown=False)):
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f'Подпапка {dir_path} удалена')


def ocr_task(img_path: str) -> tuple[int, list[tuple[str, float]]]:
    "Given image path, return (image_id, [(text, confidence)])"
    img_id = int(Path(img_path).stem)

    annotations = ocrmac.OCR(
        img_path, language_preference=["ru-RU", "en-US"]
    ).recognize()
    return img_id, [(i[0], i[1]) for i in annotations]


def process_images(images_path: str) -> pd.DataFrame:
    """
    Apply ocr_task to all images in a dataset_path, returns DF.
    Uses maximum threads.
    """
    img_paths = glob.glob(f"{images_path}/*.jpg")
    print(images_path)
    # might rewrite it to not collect all results
    # but save them every split_size
    # but 100k rows is about 10MB, not large
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(ocr_task, img_paths), total=len(img_paths)))
    return pd.DataFrame(results, columns=["image", "annotations"])