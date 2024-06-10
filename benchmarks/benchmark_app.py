import os
import random

from locust import HttpUser, TaskSet, between, task

IMAGE_DIR = "./data/real_estate_images_clean/"
image_files = [
    os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")
]


class ImageEnhancementTaskSet(TaskSet):
    @task
    def enhance_image(self):
        image_path = random.choice(image_files)
        with open(image_path, "rb") as image_file:
            files = {"image": (os.path.basename(image_path), image_file, "image/jpeg")}
            with self.client.post(
                "/enhance_image", files=files, catch_response=True
            ) as response:
                if response.status_code == 200:
                    response.success()
                elif response.status_code == 204:
                    response.success()
                else:
                    response.failure(f"Failed with status {response.status_code}")


class ImageEnhancementUser(HttpUser):
    tasks = [ImageEnhancementTaskSet]
    wait_time = between(0.3, 1.3)


if __name__ == "__main__":
    from locust import run_single_user

    run_single_user(ImageEnhancementUser)
