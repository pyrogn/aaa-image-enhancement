import argparse
import asyncio
import os
import time

import aiofiles
import httpx
import numpy as np
from tqdm.asyncio import tqdm


async def send_request(client, image_path, enhance_url):
    global total_images, no_enhancement_count, error_count, enhancement_count
    start_time = time.time()
    async with aiofiles.open(image_path, "rb") as file:
        image_data = await file.read()
        files = {"image": (os.path.basename(image_path), image_data, "image/jpeg")}
        response = await client.post(enhance_url, files=files)
        response_time = time.time() - start_time
        response_times.append(response_time)
        total_images += 1
        if response.status_code == 200:
            enhancement_count += 1
        elif response.status_code == 204:
            no_enhancement_count += 1
        elif response.status_code >= 500:
            error_count += 1


async def main(host, rps):
    enhance_url = f"http://{host}:8000/enhance_image"
    image_files = [
        os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")
    ][:100]

    start_time = time.time()

    async with httpx.AsyncClient() as client:
        for image_path in tqdm(image_files, desc="Processing images"):
            asyncio.create_task(send_request(client, image_path, enhance_url))
            await asyncio.sleep(1 / rps)

    elapsed_time = time.time() - start_time
    actual_rps = total_images / elapsed_time

    response_times_np = np.array(response_times)
    avg_response_time = np.mean(response_times_np)
    percentiles = np.percentile(response_times_np, [95, 99, 99.9])

    print(f"theoretical RPS: {rps}")
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
    parser = argparse.ArgumentParser(description="Benchmark image enhancement service.")
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="Address of the host to benchmark.",
        default="51.250.19.218",
    )
    parser.add_argument("--rps", type=int, default=10, help="Requests per second.")
    args = parser.parse_args()

    host = args.host
    rps = args.rps

    # Initialize global variables
    total_images = 0
    no_enhancement_count = 0
    error_count = 0
    enhancement_count = 0
    response_times = []

    IMAGE_DIR = "./data/real_estate_images_clean/"

    asyncio.run(main(host, rps))
