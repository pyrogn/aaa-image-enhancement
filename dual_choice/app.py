"""AB testing image enhancements."""

# beauty:
# progress bar
# fix percentages
# add extra images for fun
# add ddos protection (use limits)

import os
import random
from contextlib import asynccontextmanager

import psycopg2
import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

redis_client = redis.Redis(host="localhost", port=6379, db=0)


def get_user_pairs(user_id):
    return redis_client.lrange(user_id, 0, -1)


def add_user_pair(user_id, pair):
    redis_client.rpush(user_id, pair)


def remove_user_pair(user_id, pair):
    redis_client.lrem(user_id, 0, pair)


DATABASE_URL = "postgresql://user:password@localhost/database"


def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn


def execute_sql(query, params):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    conn.close()


def init_db():
    execute_sql(
        """
        CREATE TABLE IF NOT EXISTS image_selections (
            user_id TEXT,
            image_id INTEGER,
            selected_id INTEGER,
            other_id INTEGER,
            PRIMARY KEY (user_id, image_id, selected_id, other_id)
        )
    """,
        [],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database
    init_db()
    yield


app = FastAPI(lifespan=lifespan)


# Adjust the templates directory path
templates = Jinja2Templates(directory="dual_choice/templates")

# New data directory path
data_directory = "dual_choice/data"

# Mount the 'data' directory as a static directory
app.mount("/data", StaticFiles(directory=data_directory), name="data")


def generate_image_pairs():
    """Generate image pairs based on data, which will be random each time."""
    folders = [f.path for f in os.scandir("path/to/data") if f.is_dir()]
    pairs = []
    random.shuffle(folders)
    for folder in folders:
        images = [os.path.join(folder, img) for img in os.listdir(folder)]
        if len(images) >= 2:
            random.shuffle(images)
            pairs.extend(
                [
                    (folder, images[i], images[i + 1])
                    for i in range(0, len(images) - 1, 2)
                ]
            )
    return pairs


def load_new_images(data_directory):
    folders = [f.path for f in os.scandir(data_directory) if f.is_dir()]
    selected_folder = random.choice(folders)
    images = [os.path.join(selected_folder, img) for img in os.listdir(selected_folder)]
    selected_images = random.sample(images, 2)
    web_accessible_images = [
        img.replace(data_directory, "/data") for img in selected_images
    ]
    return web_accessible_images


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    image_pairs = generate_image_pairs()
    if not image_pairs:
        raise HTTPException(status_code=404, detail="No image pairs available")
    selected_pair = random.choice(image_pairs)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "images": [selected_pair[1], selected_pair[2]]},
    )


class ImageSelection(BaseModel):
    imageId: int
    selectedSubId: int
    nonSelectedSubId: int


@app.post("/")
async def save_selection(request: Request, selection: ImageSelection):
    assert request.client
    user_id = f"{request.client.host}_{request.headers.get('User-Agent')}"
    execute_sql(
        """
        INSERT INTO image_selections (user_id, image_id, selected_id, other_id)
        VALUES (%s, %s, %s, %s)
    """,
        (
            user_id,
            selection.imageId,
            selection.selectedSubId,
            selection.nonSelectedSubId,
        ),
    )
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
        SELECT
            SUM(CASE WHEN selected_id = ? AND other_id = ? THEN 1 ELSE 0 END)
                AS selected_count,
            COUNT(1) AS total_count
        FROM image_selections
        WHERE image_id = ? AND (
            (selected_id = ? AND other_id = ?) OR
            (selected_id = ? AND other_id = ?)
        )
    """,
        (
            selected_sub_id,
            non_selected_sub_id,
            image_id,
            selected_sub_id,
            non_selected_sub_id,
            non_selected_sub_id,
            selected_sub_id,
        ),
    )
    row = cursor.fetchone()
    if row:
        prop_selected = row["selected_count"] / row["total_count"]  # type: ignore
    else:
        prop_selected = 0
    conn.close()
    return {"prop_selected": prop_selected}


@app.get("/new-images")
async def get_new_images():
    image_pairs = generate_image_pairs()
    if not image_pairs:
        raise HTTPException(status_code=404, detail="No new images available")
    selected_pair = random.choice(image_pairs)
    return {"images": [selected_pair[1], selected_pair[2]]}
