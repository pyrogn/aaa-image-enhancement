from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import glob
import random
import os
from fastapi.staticfiles import StaticFiles

# Mount the 'data' directory as a static directory
app = FastAPI()

# templates = Jinja2Templates(directory="templates")
# awful, but won't work with previous line
current_file_path = os.path.abspath(os.path.dirname(__file__))
templates_directory = os.path.join(current_file_path, "templates")
templates = Jinja2Templates(directory=templates_directory)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    image_paths = glob.glob("./data/real_estate_images/*.jpg")
    selected_images = random.sample(image_paths, 2)
    return templates.TemplateResponse(
        "index.html", {"request": request, "images": selected_images}
    )


# запутаешься здесь
app.mount("/data", StaticFiles(directory="./data"), name="data")
