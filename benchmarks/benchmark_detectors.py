import glob
import timeit

import cv2
import numpy as np
from aaa_image_enhancement.defects_detection_fns import classical_detectors
from aaa_image_enhancement.image_utils import ImageConversions


def load_random_images(num_images=10):
    image_paths = glob.glob("./real_estate_images/*.jpg")
    random_paths = np.random.choice(image_paths, size=num_images, replace=False)
    return [ImageConversions(cv2.imread(path)) for path in random_paths]


def benchmark_detector(detector, images):
    def wrapper():
        for image in images:
            detector(image)

    return timeit.timeit(wrapper, number=1)


if __name__ == "__main__":
    num_images = 20
    images = load_random_images(num_images)

    print(f"Benchmarking detection functions on {num_images} random images:")
    results = {}
    for detector in classical_detectors:
        times = [benchmark_detector(detector, images) for _ in range(num_images)]
        results[detector.__name__] = {
            "mean": np.mean(times),
            "median": np.median(times),
            "5%": np.percentile(times, 5),
            "95%": np.percentile(times, 95),
        }

    for detector_name, metrics in results.items():
        print(f"{detector_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f} seconds")

# sample output
# ❯ py -m benchmarks.benchmark_detectors
# Benchmarking detection functions on 20 random images:
# is_noisy:
#   mean: 0.2275 seconds
#   median: 0.2253 seconds
#   5%: 0.2232 seconds
#   95%: 0.2312 seconds
# is_blurry:
#   mean: 0.0544 seconds
#   median: 0.0542 seconds
#   5%: 0.0537 seconds
#   95%: 0.0557 seconds
# is_low_light:
#   mean: 0.0096 seconds
#   median: 0.0096 seconds
#   5%: 0.0094 seconds
#   95%: 0.0098 seconds
# is_low_contrast:
#   mean: 0.2586 seconds
#   median: 0.2586 seconds
#   5%: 0.2571 seconds
#   95%: 0.2602 seconds
# is_poor_white_balance:
#   mean: 0.0376 seconds
#   median: 0.0374 seconds
#   5%: 0.0368 seconds
#   95%: 0.0385 seconds
