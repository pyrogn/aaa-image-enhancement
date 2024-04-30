import os
from src.utils import process_images
import ast
import pandas as pd

LEN_THRESHOLD = 6
DATA_PATH = "data/real_estate_images"
OUTPUT_PATH = "src/ocr"


if __name__ == "__main__":
    working_dir = os.getenv("WORKING_DIR")
    # raw_ocr = process_images(os.path.join(working_dir, DATA_PATH))
    # print(os.path.join(working_dir, DATA_PATH))
    # raw_ocr.to_csv(os.path.join(working_dir, OUTPUT_PATH, "raw_ocr.csv"))
    raw_ocr = pd.read_csv(os.path.join(working_dir, OUTPUT_PATH, "raw_ocr.csv"))
    raw_ocr.annotations = raw_ocr.annotations.apply(ast.literal_eval)

    raw_ocr["len"] = raw_ocr.annotations.apply(len)

    processed_ocr = raw_ocr[raw_ocr["len"] <= LEN_THRESHOLD]

    processed_ocr.to_csv(os.path.join(os.getenv("WORKING_DIR"), OUTPUT_PATH,"processed_ocr.csv"))
