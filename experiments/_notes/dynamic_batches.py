# Это полный копипаст чат-бота
# Но важно зафиксировать проблему и принцип решения
# с одной стороны, у нас есть ограничение в 1 секунду, с другой -
# надо эффективно масштабировать, то есть обрабатывать батчами
# будет полезно использовать https://pytorch.org/serve/

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import torch
from torch import nn
import uvicorn
from typing import List
import threading
import time
from queue import Queue

app = FastAPI()


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        # Define your model architecture here
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv1(x)


# Load the model
model = ImageModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Queue to hold incoming images
image_queue = Queue()
batch_size = 32
batch_timeout = 5  # seconds


def batch_processor():
    while True:
        batch = []
        start_time = time.time()
        while len(batch) < batch_size and (time.time() - start_time) < batch_timeout:
            try:
                image = image_queue.get(
                    timeout=batch_timeout - (time.time() - start_time)
                )
                batch.append(image)
            except:
                break  # Timeout reached with a partial batch

        if batch:
            input_tensor = torch.stack(batch)
            with torch.no_grad():
                prediction = model(input_tensor)
            print("Processed a batch of size:", len(batch))


# Start the batch processing thread
threading.Thread(target=batch_processor, daemon=True).start()


def prepare_image(file: bytes) -> torch.Tensor:  # type: ignore
    # Implement image preprocessing here
    pass


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = prepare_image(await file.read())
    image_queue.put(image)
    return {"status": "Image received and queued for processing"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
