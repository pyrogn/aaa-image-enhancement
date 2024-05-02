"""AB testing image enhancements."""

# todo:
# manage user choices and what's been completed
# better sql management (less boilerplate)
# how to read and export data
# add user id to sql (ip+useragent)
# fix bug with image id (same one)

# beauty:
# progress bar
# fix percentages
# add extra images for fun

import os
import random
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
DATABASE = "selections.db"


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS image_selections (
            image_id INTEGER,
            selected_sub_id INTEGER,
            non_selected_sub_id INTEGER,
            selection_count INTEGER DEFAULT 1,
            PRIMARY KEY (image_id, selected_sub_id, non_selected_sub_id)
        )
    """)
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database
    init_db()
    yield


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore


# Adjust the templates directory path
templates = Jinja2Templates(directory="dual_choice/templates")

# New data directory path
data_directory = "dual_choice/data"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # List all subdirectories in the data directory
    folders = [f.path for f in os.scandir(data_directory) if f.is_dir()]
    selected_folder = random.choice(folders)

    # List all images in the selected folder
    images = [os.path.join(selected_folder, img) for img in os.listdir(selected_folder)]

    # Randomly select two images without replacement
    selected_images = random.sample(images, 2)

    # Adjust the image paths for web access
    web_accessible_images = [
        img.replace(data_directory, "/data") for img in selected_images
    ]

    return templates.TemplateResponse(
        "index.html", {"request": request, "images": web_accessible_images}
    )


# Mount the 'data' directory as a static directory
app.mount("/data", StaticFiles(directory=data_directory), name="data")


class ImageSelection(BaseModel):
    imageId: int
    selectedSubId: int
    nonSelectedSubId: int


# @app.post("/", dependencies=[Depends(limiter.limit("5/second"))])
@app.post("/")
async def save_selection(request: Request, selection: ImageSelection):
    print(selection)
    # extract ip and useragent from request
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO image_selections (image_id, selected_sub_id, non_selected_sub_id)
        VALUES (?, ?, ?)
        ON CONFLICT(image_id, selected_sub_id, non_selected_sub_id)
        DO UPDATE SET selection_count = selection_count + 1
    """,
        (selection.imageId, selection.selectedSubId, selection.nonSelectedSubId),
    )
    conn.commit()
    conn.close()
    return {"message": "Selection saved"}


# переписать на долю и сделать корректный подсчет
# может оптимизировать хранение
@app.get("/selections/{image_id}/{selected_sub_id}/{non_selected_sub_id}")
async def get_selection_count(
    image_id: int, selected_sub_id: int, non_selected_sub_id: int
):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT selection_count FROM image_selections
        WHERE image_id = ? AND selected_sub_id = ? AND non_selected_sub_id = ?
    """,
        (image_id, selected_sub_id, non_selected_sub_id),
    )
    row = cursor.fetchone()
    conn.close()
    return {"count": row["selection_count"] if row else 0}


@app.get("/new-images")
async def get_new_images():
    # Similar logic to your existing image selection
    folders = [f.path for f in os.scandir(data_directory) if f.is_dir()]
    selected_folder = random.choice(folders)
    images = [os.path.join(selected_folder, img) for img in os.listdir(selected_folder)]
    selected_images = random.sample(images, 2)
    web_accessible_images = [
        img.replace(data_directory, "/data") for img in selected_images
    ]
    return {"images": web_accessible_images}
