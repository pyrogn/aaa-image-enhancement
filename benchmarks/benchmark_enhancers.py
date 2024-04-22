import glob
import timeit

import cv2
import numpy as np
from aaa_image_enhancement.enhancement_fns import classical_enhancement_fns
from tqdm import tqdm


def load_random_images(num_images=10):
    image_paths = glob.glob("./real_estate_images/*.jpg")
    random_paths = np.random.choice(image_paths, size=num_images, replace=False)
    return [cv2.imread(path) for path in random_paths]


def benchmark_enhancer(enhancer, images):
    def wrapper():
        for image in images:
            enhancer(image)

    return timeit.timeit(wrapper, number=1)


if __name__ == "__main__":
    num_images = 3
    images = load_random_images(num_images)

    print(f"Benchmarking enhancement functions on {num_images} random images:")
    results = {}
    for enhancer in tqdm(classical_enhancement_fns, desc="Enhancers"):
        times = []
        for _ in tqdm(range(num_images), desc=f"{enhancer.__name__}", leave=False):
            times.append(benchmark_enhancer(enhancer, images))
        results[enhancer.__name__] = {
            "mean": np.mean(times),
            "median": np.median(times),
            "5%": np.percentile(times, 5),
            "95%": np.percentile(times, 95),
        }

    for enhancer_name, metrics in results.items():
        print(f"{enhancer_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f} seconds")

# sample output
# ❯ py -m benchmarks.benchmark_enhancers
# Benchmarking enhancement functions on 3 random images:
# Enhancers: 100%
# deblur_image:
#   mean: 0.0071 seconds
#   median: 0.0071 seconds
#   5%: 0.0071 seconds
#   95%: 0.0073 seconds
# dehaze_image:
#   mean: 4.0650 seconds
#   median: 4.0662 seconds
#   5%: 4.0625 seconds
#   95%: 4.0666 seconds
# enhance_wb_image:
#   mean: 0.0509 seconds
#   median: 0.0191 seconds
#   5%: 0.0189 seconds
#   95%: 0.1052 seconds
# enhance_low_light:
#   mean: 89.9571 seconds
#   median: 89.7260 seconds
#   5%: 88.6169 seconds
#   95%: 91.4589 seconds
