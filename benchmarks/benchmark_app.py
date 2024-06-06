import asyncio
import os
import time

import aiofiles
import httpx
import numpy as np
from tqdm.asyncio import tqdm

host = "51.250.19.218"
# host = "localhost"

IMAGE_DIR = "./data/real_estate_images_clean/"
ENHANCE_URL = f"http://{host}:8000/enhance_image"
RPS = 10  # Requests per second

# Statistics
total_images = 0
no_enhancement_count = 0
error_count = 0
enhancement_count = 0
response_times = []


async def send_request(client, image_path):
    global total_images, no_enhancement_count, error_count, enhancement_count
    start_time = time.time()
    async with aiofiles.open(image_path, "rb") as file:
        image_data = await file.read()
        files = {"image": (os.path.basename(image_path), image_data, "image/jpeg")}
        # 200 or 204 - doesn't matter here
        response = await client.post(ENHANCE_URL, files=files)
        response_time = time.time() - start_time
        response_times.append(response_time)
        total_images += 1
        if response.status_code == 200:
            enhancement_count += 1
        elif response.status_code == 204:
            no_enhancement_count += 1
        elif response.status_code >= 500:
            error_count += 1


async def main():
    image_files = [
        os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")
    ][:100]
    # take a subset of images here!

    start_time = time.time()

    async with httpx.AsyncClient() as client:
        for image_path in tqdm(image_files, desc="Processing images"):
            asyncio.create_task(send_request(client, image_path))
            await asyncio.sleep(1 / RPS)

    elapsed_time = time.time() - start_time
    actual_rps = total_images / elapsed_time

    response_times_np = np.array(response_times)
    avg_response_time = np.mean(response_times_np)
    percentiles = np.percentile(response_times_np, [95, 99, 99.9])

    print(f"theoretical RPS: {RPS}")
    print(f"actual RPS: {actual_rps:.2f}")
    print(f"Total images processed: {total_images}")
    print(f"Enhancements: {enhancement_count}")
    print(f"No enhancements needed: {no_enhancement_count}")
    print(f"Errors: {error_count}")
    print(f"Average response time: {avg_response_time:.4f} seconds")
    print(f"95th percentile response time: {percentiles[0]:.4f} seconds")
    print(f"99th percentile response time: {percentiles[1]:.4f} seconds")
    print(f"99.9th percentile response time: {percentiles[2]:.4f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
