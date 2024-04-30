from src.utils import process_images_directories
from os import getenv
DATA_PATH = "data/real_estate_images/"

if __name__ == "__main__":
    working_dir = getenv("WORKING_DIR")
    process_images_directories(data_path=DATA_PATH)
